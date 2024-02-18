#"为了测试cotta在所有模型上的效果，将cotta单独封装为一个类，可以嵌套任何模型"

from utils import split_last, merge_last,Preprocess4Sample_t,Preprocess4Rotation_t,Preprocess4Normalization_t, Preprocess4Noise_t, Preprocess4Permute_t
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torchvision.transforms as transforms
import torch.jit
from torchattacks import FGSM, PGD, MIFGSM, CW, DeepFool
# from utils import Preprocess4Normalization, Preprocess4Sample, Preprocess4Rotation, Preprocess4Noise, Preprocess4Permute
# pipeline_tta = [Preprocess4Normalization(args.input),Preprocess4Sample(args.seq_len, temporal=0.4)
#             , Preprocess4Rotation(), Preprocess4Noise(), Preprocess4Permute()]
# data_transform = transforms.Compose([
#     transforms.Lambda(lambda x: time_shift(x, 2)),
#     transforms.Lambda(lambda x: time_scale(x, 0.5))
# ])
# pipeline = [ Preprocess4Sample(seq_len, temporal=0.4)
#             , Preprocess4Rotation(), Preprocess4Mask(args.mask_cfg, full_sequence=True)]
torch.autograd.set_detect_anomaly(True)
def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
        # return x


class Embeddings(nn.Module):

    def __init__(self, cfg, pos_embed=None):
        super().__init__()

        # factorized embedding
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden) # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)

        # factorized embedding
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)


class MultiProjection(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        return q, k, v


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        #scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        h = self.embed(x)

        for _ in range(self.n_layers):
            # h = block(h, mask)
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class LIMUBertModel4Pretrain(nn.Module):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = Transformer(cfg) # encoder
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ = gelu
        self.norm = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        return logits_lm


class ClassifierLSTM(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('lstm' + str(i), nn.LSTM(input, cfg.rnn_io[i][1], num_layers=cfg.num_layers[i], batch_first=True))
            else:
                self.__setattr__('lstm' + str(i),
                                 nn.LSTM(cfg.rnn_io[i][0], cfg.rnn_io[i][1], num_layers=cfg.num_layers[i],
                                         batch_first=True))
            self.__setattr__('bn' + str(i), nn.BatchNorm1d(cfg.seq_len))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_rnn):
            lstm = self.__getattr__('lstm' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h, _ = lstm(h)
            if self.activ:
                h = F.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierGRU(nn.Module):
    def __init__(self, cfg, input=None, output=None, feats=False):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('gru' + str(i), nn.GRU(input, cfg.rnn_io[i][1], num_layers=cfg.num_layers[i], batch_first=True))
            else:
                self.__setattr__('gru' + str(i),
                                 nn.GRU(cfg.rnn_io[i][0], cfg.rnn_io[i][1], num_layers=cfg.num_layers[i],
                                         batch_first=True))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_rnn):
            rnn = self.__getattr__('gru' + str(i))
            h, _ = rnn(h)
            if self.activ:
                h = F.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierAttn(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.embd = nn.Embedding(cfg.seq_len, input)
        self.proj_q = nn.Linear(input, cfg.atten_hidden)
        self.proj_k = nn.Linear(input, cfg.atten_hidden)
        self.proj_v = nn.Linear(input, cfg.atten_hidden)
        self.attn = nn.MultiheadAttention(cfg.atten_hidden, cfg.num_head)
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.flatten = nn.Flatten()
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        seq_len = input_seqs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=input_seqs.device)
        pos = pos.unsqueeze(0).expand(input_seqs.size(0), seq_len)  # (S,) -> (B, S)
        h = input_seqs + self.embd(pos)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        h, weights = self.attn(q, k, v)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            if i == self.num_linear - 1:
                h = self.flatten(h)
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierCNN2D(nn.Module):
    def __init__(self, cfg, output=None):
        super().__init__()
        for i in range(cfg.num_cnn):
            if i == 0:
                self.__setattr__('cnn' + str(i), nn.Conv2d(1, cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            else:
                self.__setattr__('cnn' + str(i), nn.Conv2d(cfg.conv_io[i][0], cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            self.__setattr__('bn' + str(i), nn.BatchNorm2d(cfg.conv_io[i][1]))
        self.pool = nn.MaxPool2d(cfg.pool[0], stride=cfg.pool[1], padding=cfg.pool[2])
        self.flatten = nn.Flatten()
        for i in range(cfg.num_linear):
            if i == 0:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.flat_num, cfg.linear_io[i][1]))
            elif output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_cnn = cfg.num_cnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs.unsqueeze(1)
        for i in range(self.num_cnn):
            cnn = self.__getattr__('cnn' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h = cnn(h)
            if self.activ:
                h = F.relu(h)
            h = bn(self.pool(h))
            # h = self.pool(h)
        h = self.flatten(h)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierCNN1D(nn.Module):
    def __init__(self, cfg, output=None):
        super().__init__()
        for i in range(cfg.num_cnn):
            if i == 0:
                self.__setattr__('cnn' + str(i),
                                 nn.Conv1d(cfg.seq_len, cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            else:
                self.__setattr__('cnn' + str(i),
                                 nn.Conv1d(cfg.conv_io[i][0], cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            self.__setattr__('bn' + str(i), nn.BatchNorm1d(cfg.conv_io[i][1]))
        self.pool = nn.MaxPool1d(cfg.pool[0], stride=cfg.pool[1], padding=cfg.pool[2])
        self.flatten = nn.Flatten()
        for i in range(cfg.num_linear):
            if i == 0:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.flat_num, cfg.linear_io[i][1]))
            elif output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_cnn = cfg.num_cnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_cnn):
            cnn = self.__getattr__('cnn' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h = cnn(h)
            if self.activ:
                h = F.relu(h)
            h = self.pool(h)
            # h = bn(h)
            # h = self.pool(h)
        h = self.flatten(h)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class BERTClassifier(nn.Module):

    def __init__(self, bert_cfg, classifier=None, frozen_bert=False):
        super().__init__()
        self.transformer = Transformer(bert_cfg)
        if frozen_bert:
            for p in self.transformer.parameters():
                p.requires_grad = False
        self.classifier = classifier

    def forward(self, input_seqs, training=False): #, training
        h = self.transformer(input_seqs)
        h = self.classifier(h, training)
        return h

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)
def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model
class BERTClassifierTTA(nn.Module):
    
    def __init__(self, bert_cfg, classifier=None, frozen_bert=False, optimizer=None):
        super().__init__()
        self.transformer = Transformer(bert_cfg)
        if frozen_bert:
            for p in self.transformer.parameters():
                p.requires_grad = False
        self.classifier = classifier
        self.episodic = False
        self.steps = 1
        self.optimizer = optimizer
        self.transform = data_transform
    # def forward(self, input_seqs, training=False): #, training
    #     h = self.transformer(input_seqs)
    #     h = self.classifier(h, training)
    #     return h
    def forward(self, input_seqs, training=False):
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.classifier, self.optimizer)
        if self.episodic:
            self.reset()
        # input_seqs=data_transform(input_seqs)
        h = self.transformer(input_seqs)
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(h, self.classifier, self.optimizer)
        # h = self.classifier(h, training)
        return outputs
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        # load_model_and_optimizer(self.model, self.optimizer,
        #                          self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        outputs = model(x)
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = 64
        outputs_emas = []
        for i in range(N):
            # outputs_  = self.model_ema(self.transform(x)).detach() #这个地方有缺陷，理论上要添加许多把不同的transform
            outputs_  = self.model_ema(x).detach()
            outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if anchor_prob.mean(0)<0.9:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        # outputs_ema = standard_ema
        # Student update
        loss = (softmax_entropy(outputs, outputs_ema)).mean(0) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.classifier, alpha_teacher=0.5) #这地方需要调参
        # Stochastic restore
        if True:
            for nm, m  in self.classifier.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<0.1).float().cuda()  #self.rst超参
                        with torch.no_grad():
                            # print(p.device,mask.device, self.model_state.device)
                            p.data = self.model_state[f"{nm}.{npp}"].cuda() * mask + p * (1.-mask)
        return outputs_ema
class BenchmarkDCNN(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, (5, 1))
        self.bn1 = nn.BatchNorm2d(50)
        self.conv2 = nn.Conv2d(50, 40, (5, 1))
        self.bn2 = nn.BatchNorm2d(40)
        # print("dcnn")
        if cfg.seq_len <= 20:
            self.conv3 = nn.Conv2d(40, 20, (2, 1))
        else:
            self.conv3 = nn.Conv2d(40, 20, (3, 1))
        self.bn3 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d((2, 1))
        self.lin1 = nn.Linear(156, 400) # input * cfg.flat_num
        
        self.lin2 = nn.Linear(400, output)

    def forward(self, input_seqs, training=False):
        # print(input_seqs.device)
        h = input_seqs.unsqueeze(1)
        h = F.relu(F.tanh(self.conv1(h)))
        h = self.bn1(self.pool(h))
        h = F.relu(F.tanh(self.conv2(h)))
        h = self.bn2(self.pool(h))
        h = F.relu(F.tanh(self.conv3(h)))
        h = h.view(h.size(0), h.size(1), h.size(2) * h.size(3))
        h = self.lin1(h)
        h = F.relu(F.tanh(torch.sum(h, dim=1)))
        h = self.normalize(h[:, :, None, None])
        h = self.lin2(h[:, :, 0, 0])
        return h

    def normalize(self, x, k=1, alpha=2e-4, beta=0.75):
        # x = x.view(x.size(0), x.size(1) // 5, 5, x.size(2), x.size(3))#
        # y = x.clone()
        # for s in range(x.size(0)):
        #     for j in range(x.size(1)):
        #         for i in range(5):
        #             norm = alpha * torch.sum(torch.square(y[s, j, i, :, :])) + k
        #             norm = torch.pow(norm, -beta)
        #             x[s, j, i, :, :] = y[s, j, i, :, :] * norm
        # x = x.view(x.size(0), x.size(1) * 5, x.size(3), x.size(4))
        return x


class BenchmarkDeepSense(nn.Module):

    def __init__(self, cfg, input=None, output=None, num_filter=8):
        super().__init__()
        self.sensor_num = input // 3
        for i in range(self.sensor_num):
            self.__setattr__('conv' + str(i) + "_1", nn.Conv2d(1, num_filter, (2, 3)))
            self.__setattr__('conv' + str(i) + "_2", nn.Conv2d(num_filter, num_filter, (3, 1)))
            self.__setattr__('conv' + str(i) + "_3", nn.Conv2d(num_filter, num_filter, (2, 1)))
            self.__setattr__('bn' + str(i) + "_1", nn.BatchNorm2d(num_filter))
            self.__setattr__('bn' + str(i) + "_2", nn.BatchNorm2d(num_filter))
            self.__setattr__('bn' + str(i) + "_3", nn.BatchNorm2d(num_filter))
        self.conv1 = nn.Conv2d(1, num_filter, (2, self.sensor_num))
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = nn.Conv2d(num_filter, num_filter, (3, 1))
        self.bn2 = nn.BatchNorm2d(num_filter)
        self.conv3 = nn.Conv2d(num_filter, num_filter, (2, 1))
        self.bn3 = nn.BatchNorm2d(num_filter)
        self.flatten = nn.Flatten()

        self.lin1 = nn.Linear(cfg.flat_num, 12)
        self.lin2 = nn.Linear(12, output)


    def forward(self, input_seqs, training=False):
        h = input_seqs.view(input_seqs.size(0), input_seqs.size(1), self.sensor_num, 3)
        hs = []
        for i in range(self.sensor_num):
            t = h[:, :, i, :]
            t = torch.unsqueeze(t, 1)
            for j in range(3):
                cv = self.__getattr__('conv' + str(i) + "_" + str(j + 1))
                bn = self.__getattr__('bn' + str(i) + "_" + str(j + 1))
                t = bn(F.relu(cv(t)))
            hs.append(self.flatten(t)[:, :, None])
        h = torch.cat(hs, dim=2)
        h = h.unsqueeze(1)
        h = self.bn1(F.relu(self.conv1(h)))
        h = self.bn2(F.relu(self.conv2(h)))
        h = self.bn3(F.relu(self.conv3(h)))
        h = self.flatten(h)
        h = self.lin2(F.relu(self.lin1(h)))
        return h


class BenchmarkTPNPretrain(nn.Module):
    def __init__(self, cfg, task_num, input=None):
        super().__init__()
        self.conv1 = nn.Conv1d(input, 32, kernel_size=6)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4)
        self.conv3 = nn.Conv1d(64, 96, kernel_size=2)
        self.flatten = nn.Flatten()
        for i in range(task_num):
            self.__setattr__('slin' + str(i) + "_1", nn.Linear(96, 256))
            self.__setattr__('slin' + str(i) + "_2", nn.Linear(256, 1))
        self.task_num = task_num

    def forward(self, input_seqs, training=False):
        h = input_seqs.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = F.dropout(h, p=0.1, training=training)
        h = F.relu(self.conv2(h))
        h = F.dropout(h, p=0.1, training=training)
        h = F.relu(self.conv3(h))
        h = F.dropout(h, p=0.1, training=training)
        h = self.flatten(torch.max(h, 2)[0])
        hs = []
        for i in range(self.task_num):
            lin1 = self.__getattr__('slin' + str(i) + "_1")
            lin2 = self.__getattr__('slin' + str(i) + "_2")
            hl = F.relu(lin1(h))
            hl = F.sigmoid(lin2(hl))
            hs.append(hl)
        hf = torch.stack(hs)[:, :, 0]
        hf = torch.transpose(hf, 0, 1)
        return hf


class BenchmarkTPNClassifier(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.conv1 = nn.Conv1d(input, 32, kernel_size=6)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4)
        self.conv3 = nn.Conv1d(64, 96, kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(96, 1024)
        self.fc2 = nn.Linear(1024, output)
        for p in self.conv1.parameters():
            p.requires_grad = False
        for p in self.conv2.parameters():
            p.requires_grad = False
        for p in self.conv3.parameters():
            p.requires_grad = False

    def forward(self, input_seqs, training=False):
        h = input_seqs.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = F.dropout(h, p=0.1, training=training)
        h = F.relu(self.conv2(h))
        h = F.dropout(h, p=0.1, training=training)
        h = F.relu(self.conv3(h))
        h = F.dropout(h, p=0.1, training=training)
        h = self.flatten(torch.max(h, 2)[0])
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return h

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        # self.args = args
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state,_,_ = \
            copy_model_and_optimizer(self.model, self.optimizer)
    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        model.train()
        """Forward and adapt model on batch of data.

        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        outputs = model(x)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        # adapt

        
        loss = softmax_entropy_tent(outputs).mean(0)
        # print(loss)
        # loss, _ = marginal_entropy(outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            outputs = model(x)
        return outputs


class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9, arg=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        # self.transform = [Preprocess4Normalization(arg.input),Preprocess4Sample(arg.seq_len, temporal=0.4)
        #     , Preprocess4Rotation(), Preprocess4Noise(), Preprocess4Permute()]
        self.transform = transforms.Compose([Preprocess4Normalization_t(arg.input),Preprocess4Sample_t(arg.seq_len, temporal=0.4)
                          ,Preprocess4Rotation_t(), Preprocess4Noise_t(), Preprocess4Permute_t()]) # Preprocess4Normalization(arg.sr),
        # self.transform = transforms.Compose([Preprocess4Normalization(arg.sr)])
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

    def forward_tta(self, x, x_ema):
        # print('x:',x.shape)
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer,x_ema)

        return outputs
    def forward(self, x):
        return self.model(x)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer,x_ema):
        # print(x.shape)
        # print(x[0].shape)
        # print(len(x))
        outputs = self.model(x)
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = 32 
        outputs_emas = []
        for i in range(N):
            # for t in range(len(x)):
            #     x[t]=self.transform(x[t])
            # print(x.shape)
            # outputs_  = self.model_ema(self.transform(x)).detach()
            
            "here is very important"
            # print(i)
            # for proc in self.transform:
            #     x = proc(x)
            # print(self.transform(x).shape)
            # outputs_  = self.model_ema(x_ema).detach()
            outputs_  = self.model_ema(x).detach()
            outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if anchor_prob.mean(0)<self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        # Student update
        loss = (softmax_entropy(outputs, outputs_ema)).mean(0) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.rst).float().cuda() 
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"].cuda() * mask + p * (1.-mask)
        return outputs_ema


class LayerActivations:
    def __init__(self, model):
        # self.opt = opt
        self.model = model
        self.model.eval()
        self.build_hook()

    def build_hook(self):
        for name, m in self.model.named_children():
            # print(name)
            # print(m.bn3)
            if isinstance(m, torch.nn.Sequential) and name == 'bn3': # The 'bn3' is especially for the DCNN model, it may be different for other models
                print("what")
                # self.hook = m.register_forward_hook(self.hook_fn)
                self.hook = self.model.bn3.register_forward_hook(self.hook_fn)
            # print(self)
            # print(self.model)
            # print(self.model.model.bn3)
                break

    def hook_fn(self, module, input, output):
        self.features = input[0]

    def remove_hook(self):
        self.hook.remove()

    def run_hook(self,x):
        self.model(x)
        # self.remove_hook()
        return self.features

class Feature_Correlations:
    def __init__(self,POWER_list, mode='mad'):
        self.power = POWER_list
        self.mode = mode

    def train(self, in_data):
        self.in_data = in_data
        if 'mad' in self.mode:
            self.medians, self.mads = self.get_median_mad(self.in_data)
            self.mins, self.maxs = self.minmax_mad()

    def minmax_mad(self):
        mins = []
        maxs = []
        for L, mm in enumerate(zip(self.medians,self.mads)):
            medians, mads = mm[0], mm[1]
            if L==len(mins):
                mins.append([None]*len(self.power))
                maxs.append([None]*len(self.power))
            for p, P in enumerate(self.power):
                    mins[L][p] = medians[p]-mads[p]*10
                    maxs[L][p] = medians[p]+mads[p]*10
        return mins, maxs

    def G_p(self, ob, p):
        temp = ob.detach()
        temp = temp**p
        temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
        temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1))))
        temp = temp.triu()
        temp = temp.sign()*torch.abs(temp)**(1/p)
        temp = temp.reshape(temp.shape[0],-1)
        self.num_feature = temp.shape[-1]/2
        return temp

    def get_median_mad(self, feat_list):
        medians = []
        mads = []
        for L,feat_L in enumerate(feat_list):
            if L==len(medians):
                medians.append([None]*len(self.power))
                mads.append([None]*len(self.power))
            for p,P in enumerate(self.power):
                g_p = self.G_p(feat_L,P)
                current_median = g_p.median(dim=0,keepdim=True)[0]
                current_mad = torch.abs(g_p - current_median).median(dim=0,keepdim=True)[0]
                medians[L][p] = current_median
                mads[L][p] = current_mad
        return medians, mads

    def get_deviations_(self, feat_list):
        deviations = []
        batch_deviations = []
        for L,feat_L in enumerate(feat_list):
            dev = 0
            for p,P in enumerate(self.power):
                g_p = self.G_p(feat_L,P)
                dev +=  (F.relu(self.mins[L][p]-g_p)/torch.abs(self.mins[L][p]+10**-6)).sum(dim=1,keepdim=True)
                dev +=  (F.relu(g_p-self.maxs[L][p])/torch.abs(self.maxs[L][p]+10**-6)).sum(dim=1,keepdim=True)
            batch_deviations.append(dev.cpu().detach().numpy())
        batch_deviations = np.concatenate(batch_deviations,axis=1)
        deviations.append(batch_deviations)
        deviations = np.concatenate(deviations,axis=0) /self.num_feature /len(self.power)
        return deviations

    def get_deviations(self, feat_list):
        deviations = []
        batch_deviations = []
        for L,feat_L in enumerate(feat_list):
            dev = 0
            for p,P in enumerate(self.power):
                g_p = self.G_p(feat_L,P)
                dev += torch.sum(torch.abs(g_p-self.medians[L][p])/(self.mads[L][p]+1e-6),dim=1,keepdim=True)
            batch_deviations.append(dev.cpu().detach().numpy())
        batch_deviations = np.concatenate(batch_deviations,axis=1)
        deviations.append(batch_deviations)
        deviations = np.concatenate(deviations,axis=0)/self.num_feature /len(self.power)
        return deviations

def threshold_determine(clean_feature_target, ood_detection):
    test_deviations_list = []
    step = 5
    for i in range(step):
        index_mask = np.ones((len(clean_feature_target),))
        index_mask[i*int(len(clean_feature_target)//step):(i+1)*int(len(clean_feature_target)//step)] = 0
        clean_feature_target_train= clean_feature_target[np.where(index_mask == 1)]
        clean_feature_target_test = clean_feature_target[np.where(index_mask == 0)]
        ood_detection.train(in_data=[clean_feature_target_train],)
        test_deviations = ood_detection.get_deviations_([clean_feature_target_test])
        test_deviations_list.append(test_deviations)
    test_deviations = np.concatenate(test_deviations_list,0)
    test_deviations_sort = np.sort(test_deviations,0)
    percentile_95 = test_deviations_sort[int(len(test_deviations_sort)*0.95)][0]
    percentile_99 = test_deviations_sort[int(len(test_deviations_sort)*0.99)][0]
    # print(f'percentile_95:{percentile_95}')
    # print(f'percentile_99:{percentile_99}')
    # plt.hist(test_deviations)
    # plt.show()
    return percentile_95, percentile_99

# def kmmd_dist(x1, x2):
#     X_total = torch.cat([x1,x2],0)
#     Gram_matrix = gaussian_kernel(X_total,X_total,kernel_mul=2.0, kernel_num=2, fix_sigma=0, mean_sigma=0)
#     n = int(x1.shape[0])
#     m = int(x2.shape[0])
#     print(n,m)
#     print("gram",Gram_matrix)
#     x1x1 = Gram_matrix[:n, :n]
#     x2x2 = Gram_matrix[n:, n:]
#     x1x2 = Gram_matrix[:n, n:]
#     # x2x1 = Gram_matrix[n:, :n]  # Gram_matrix is symmetric
#     diff = torch.mean(x1x1) + torch.mean(x2x2) - 2 * torch.mean(x1x2)
#     diff = (m*n)/(m+n)*diff
#     return diff.to(torch.device('cpu')).numpy()

def kmmd_dist(x1, x2):
    X_total = torch.cat([x1,x2],0)
    Gram_matrix = gaussian_kernel(X_total,X_total,kernel_mul=2.0, kernel_num=2, fix_sigma=0, mean_sigma=0)
    n = int(x1.shape[0])
    m = int(x2.shape[0])
    # print(n,m)
    # print("gram",Gram_matrix)
    # x1x1 = Gram_matrix[:n, :n]
    # x2x2 = Gram_matrix[n:, n:]
    # x1x2 = Gram_matrix[:n, n:]
    # x2x1 = Gram_matrix[n:, :n]  # Gram_matrix is symmetric
    # diff = torch.mean(x1x1) + torch.mean(x2x2) - 2 * torch.mean(x1x2)
    diff = (m*n)/(m+n)*Gram_matrix
    return diff

def gaussian_kernel(x1, x2, kernel_mul=2.0, kernel_num=5, fix_sigma=0, mean_sigma=0):
    x1_sample_size = x1.shape[0]
    x2_sample_size = x2.shape[0]
    # print(x1_sample_size,x2_sample_size)
    # print(x1.shape,x2.shape)
    x1_tile_shape = []
    x2_tile_shape = []
    norm_shape = []
    for i in range(len(x1.shape) + 1):
        if i == 1:
            x1_tile_shape.append(x2_sample_size)
        else:
            x1_tile_shape.append(1)
        if i == 0:
            x2_tile_shape.append(x1_sample_size)
        else:
            x2_tile_shape.append(1)
        if not (i == 0 or i == 1):
            norm_shape.append(i)

    tile_x1 = torch.unsqueeze(x1, 1).repeat(x1_tile_shape)
    tile_x2 = torch.unsqueeze(x2, 0).repeat(x2_tile_shape)
    L2_distance = torch.square(tile_x1 - tile_x2).sum(dim=norm_shape)
    # print("l2",L2_distance)
    # bandwidth inference
    # print(L2_distance.shape)
    if fix_sigma:
        bandwidth = fix_sigma
    elif mean_sigma:
        bandwidth = torch.mean(L2_distance)
    else:  ## median distance
        bandwidth = torch.mean(L2_distance)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  #L2_distance #

def J_t(model, source, target):
    J_t = []
    def hook_fn(module, input, output):
        # print("Hook called!")
        # print("Module:", module)
        # print("Input:", input)
        # print("Output:", output)
        return
    hook = model.lin1.register_forward_hook(hook_fn)
    original_feature = []
    augmented_feature = []
    original_feature=model(source).to(torch.device('cpu'))
    augmented_feature=model(target).to(torch.device('cpu'))
    # print(original_feature[0].shape)
    feature_test = torch.cat([original_feature,augmented_feature],0)
    original_label_test = np.zeros((original_feature.shape[0],))
    augmented_label_test = np.ones((augmented_feature.shape[0],))
    label_test = np.concatenate([original_label_test,augmented_label_test],0)
    ood_detection = Feature_Correlations(POWER_list=np.arange(1,9),mode='mad')
    ood_detection.train(in_data=[original_feature])
    original_deviations_sort = np.sort(ood_detection.get_deviations_([original_feature]),0)
    augmented_deviations_sort = np.sort(ood_detection.get_deviations_([augmented_feature]),0)
    percentile_95 = np.where(augmented_deviations_sort > original_deviations_sort[int(len(original_deviations_sort)*0.95)], 1, 0)
    # print(f'percentile_95:{clean_deviations_sort[int(len(clean_deviations_sort)*0.95)]},TP95:{percentile_95.sum()/len(bd_deviations_sort)}')
    percentile_99 = np.where(augmented_deviations_sort > original_deviations_sort[int(len(original_deviations_sort)*0.99)], 1, 0)
    # print(f'percentile_99:{clean_deviations_sort[int(len(clean_deviations_sort)*0.99)]},TP99:{percentile_99.sum()/len(bd_deviations_sort)}')
    threshold_95, threshold_99 = threshold_determine(original_feature, ood_detection)
    test_deviations = ood_detection.get_deviations_([feature_test])
    ood_label_95 = np.where(test_deviations > threshold_95, 1, 0).squeeze()
    ood_label_99 = np.where(test_deviations > threshold_99, 1, 0).squeeze()
    false_negetive_95 = np.where(label_test - ood_label_95 > 0, 1, 0).squeeze()
    false_negetive_99 = np.where(label_test - ood_label_99 > 0, 1, 0).squeeze()
    false_positive_95 = np.where(label_test - ood_label_95 < 0, 1, 0).squeeze()
    false_positive_99 = np.where(label_test - ood_label_99 < 0, 1, 0).squeeze()
    clean_feature_group = feature_test[np.where(ood_label_95==0)]
    bd_feature_group = feature_test[np.where(ood_label_95==1)]
    clean_feature_flat = torch.mean(clean_feature_group,dim=1)
    bd_feature_flat = torch.mean(bd_feature_group,dim=1)
    if bd_feature_flat.shape[0] < 1:
        kmmd = torch.Tensor([0])
        kmmd.requires_grad = True
        # kmmd.grad_fn = torch.mul(kmmd, 1)
    else:
        kmmd = kmmd_dist(clean_feature_flat, bd_feature_flat)
    # J_t.append(kmmd.item())
    # J_t = np.asarray(J_t)
    # J_t_median = np.median(J_t)
    # J_MAD = np.median(np.abs(J_t - J_t_median))
    # J_star = np.abs(J_t - J_t_median)/1.4826/(J_MAD+1e-6)
    # print('J_t_median:',J_t_median)
    return kmmd

class CoTTA_attack(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9, arg=None):
        super().__init__()
        self.model = model
        self.model.to(arg.device)
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        # self.transform = [Preprocess4Normalization(arg.input),Preprocess4Sample(arg.seq_len, temporal=0.4)
        #     , Preprocess4Rotation(), Preprocess4Noise(), Preprocess4Permute()]
        self.transform = [Preprocess4Normalization_t(arg.input),Preprocess4Sample_t(arg.seq_len, temporal=0.4)
                          ,Preprocess4Rotation_t(), Preprocess4Noise_t(), Preprocess4Permute_t()]
        self.transform1_norm = transforms.Compose([Preprocess4Normalization_t(arg.input)])
        self.transform2_sample = transforms.Compose([Preprocess4Sample_t(arg.seq_len, temporal=0.4)])
        self.transform3_rot = transforms.Compose([Preprocess4Rotation_t()])
        self.transform4_noise = transforms.Compose([Preprocess4Noise_t()])
        self.transform5_permu = transforms.Compose([Preprocess4Permute_t()])
        self.attack = PGD(self.model,eps=0.04, alpha=1 / 255, steps=10) # eps=0.03, alpha=2 / 255,
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

    def forward(self, x):
        return self.model(x)
    
    def forward_tta(self, x, x_ema,epoch):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer,x_ema,epoch)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer,x_ema,epoch):
        outputs = self.model(x)
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = 32 # 32 64 
        outputs_emas = []
        inputs=[]
        for transform in self.transform:
            input = transform(x.clone())
            J_t_median = J_t(self.model, x, input)
            # print(J_t_median)
            inputs.append([input,J_t_median])
        x_attack = self.attack(x,outputs)
        x_attack = x_attack.to('cuda')
        inputs.append([x_attack,J_t(self.model, x, x_attack)])
        J_list = sorted(inputs,key=lambda x:x[1])
        # inputs = min(inputs,key=lambda x:x[1])
        inputs = J_list[round(np.exp(epoch*0.01))-1]
        if inputs[1]<np.exp(epoch*0.1): # epoch * 0.1
            # print('J_t_median:',inputs[1])
            for i in range(N):
                outputs_  = self.model_ema(inputs[0]).detach()
                "here is very important"
                outputs_emas.append(outputs_)
            # Threshold choice discussed in supplementary
            if anchor_prob.mean(0)<self.ap:
                outputs_ema = torch.stack(outputs_emas).mean(0)
            else:
                outputs_ema = standard_ema
            # Student update
            optimizer.zero_grad()
            gram_loss = inputs[1]
            gram_loss = gram_loss.to('cuda')
            print('gramloss:',gram_loss)
            loss = (softmax_entropy(outputs, outputs_ema)).mean(0)+gram_loss
            loss.backward()
            optimizer.step()
            # Teacher update
            self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        # mask = (torch.rand(p.shape)<self.rst).float().cuda() 
                        mask=0.05
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"].cuda() * mask + p * (1.-mask)
        # return outputs_ema 
        return standard_ema   

@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def softmax_entropy_tent(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor
def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model
def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
def fetch_classifier(args):
    if 'lstm' in args.model:
        model = ClassifierLSTM(args, input=args.input, output=args.output)
    elif 'gru' in args.model:
        model = ClassifierGRU(args, input=args.input, output=args.output)
    elif 'dcnn' in args.model:
        # print("dcnn")
        print(args)
        model = BenchmarkDCNN(args, input=args.input, output=args.activity_label_size)
    elif 'cnn2' in args.model:
        model = ClassifierCNN2D(args, output=args.output)
    elif 'cnn1' in args.model:
        model = ClassifierCNN1D(args, output=args.output)
    elif 'deepsense' in args.model:
        model = BenchmarkDeepSense(args, input=args.input, output=args.output)
    elif 'attn' in args.model:
        model = ClassifierAttn(args, input=args.input, output=args.output)
    else:
        model = None
    return model