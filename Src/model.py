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
from nwd import NuclearWassersteinDiscrepancy
from match_utils import ce_loss, consistency_loss, Get_Scalar, concat_all_gather
from torch.cuda.amp import autocast, GradScaler
import contextlib
from gram import J_t

from datasetPre import get_device

from torch.autograd import Function
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


class BaseModule(nn.Module):

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            # if k in state_dict:
            if k.replace('transformer', 'encoder') in state_dict:
                # state_dict.update({k: v})
                state_dict.update({k.replace('transformer','encoder'):v})
        self.load_state_dict(state_dict)

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm = LayerNorm(cfg)
        self.pred = nn.Linear(cfg.hidden, cfg.feature_num)

    def forward(self, input_seqs):
        h_masked = gelu(self.linear(input_seqs))
        h_masked = self.norm(h_masked)
        return self.pred(h_masked)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class CompositeClassifierDA(BaseModule):

    def __init__(self, ae_cfg, classifier=None, output_embed=False, freeze_encoder=False, freeze_decoder=False,
                 freeze_classifier=False):
        super().__init__()
        self.encoder = Transformer(ae_cfg)
        if freeze_encoder:
            freeze(self.encoder)
        self.decoder = Decoder(ae_cfg)
        if freeze_decoder:
            freeze(self.decoder)
        self.classifier = classifier
        if freeze_classifier:
            freeze(self.classifier)
        self.output_embed = output_embed
        self.domain_classifier = self.init_domain_classifier()

    def forward(self, input_seqs, training=False, output_clf=True, masked_pos=None, embed=False, lam=1.0):
        h = self.encoder(input_seqs)
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
            h = torch.gather(h, 1, masked_pos)
        h_features = self.classifier(h, embed=True)
        if embed:
            return h_features
        if output_clf:
            class_output = self.classifier(h, training)
            if training:
                h_reverse = ReverseLayerF.apply(h_features, lam)
                domain_output = self.domain_classifier(h_reverse)
                return class_output, domain_output
            else:
                return class_output
        else:
            r = self.decoder(h)
            return r

    def init_domain_classifier(self):
        domain_classifier = nn.Sequential()
        domain_classifier.add_module('gru_d_fc1', nn.Linear(20, 72))
        domain_classifier.add_module('gru_d_relu1', nn.ReLU())
        domain_classifier.add_module('gru_d_fc2', nn.Linear(72, 2))
        return domain_classifier


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
    def __init__(self, cfg, embed=None):
        super().__init__()
        if embed is None:
            self.embed = Embeddings(cfg)
        else:
            self.embed = embed
        
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
        self.bidirectional = any(cfg.rnn_bidirection) # yn

    def forward(self, input_seqs, training=False, embed=False):
        h = input_seqs
        for i in range(self.num_rnn):
            rnn = self.__getattr__('gru' + str(i))
            h, _ = rnn(h)
            if self.activ:
                h = F.relu(h)
        h = h[:, -1, :]
        if embed:
            return h
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


class CoTTA_attack(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9, arg=None):
        super().__init__()
        self.model = model
        self.device = get_device(arg.g)
        self.model.to(self.device)
        self.args = arg
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
        # self.intermedia_feature = LayerActivations(self.model)
        self.attack = PGD(self.model,eps=0.04, alpha=1 / 255, steps=10) # eps=0.03, alpha=2 / 255,
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

    def forward(self, x):
        # outputs = self.model(x)
        # print(outputs.device)
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
        discrepancy = NuclearWassersteinDiscrepancy(self.model)
        # intermedia_feature = LayerActivations(model)
        # print(x.shape)
        # print(x[0].shape)
        # print(len(x))
        outputs = self.model(x)
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = 32 # 32 64 
        outputs_emas = []
        # print(outputs.device)
        # print(x.device)
        # x_attack = self.transform1_norm(x_attack)
        # x_attack = self.transform2_sample(x_attack)
        # x_attack = self.transform3_rot(x_attack)
        # outputs1_  = self.model_ema(self.transform1_norm(x)).detach()
        #         outputs_emas.append(outputs1_*0.1)
        #         outputs2_  = self.model_ema(self.transform2_sample(x)).detach()
        #         # outputs_emas.append(outputs2_*0.4)
        #         outputs3_  = self.model_ema(self.transform3_rot(x)).detach()
        
        # print('J_MAD:',J_MAD)
        # print('J_star:',J_star)
        # # [print('class %d: %.2f'%(i,J_star_i)) for i,J_star_i in enumerate(J_star)]
        # [print('%.2f'%(J_star_i)) for i,J_star_i in enumerate(J_star)]
        # J_t_median = J_t(self.model, x, x_attack)
        
        # inputs_dis=[]
        # for i in np.arange(0, len(self.transform)+1, 1):
        #     print(i)
        #     if i==2:
        #         input = self.transform[i](x_ema.clone()) # x_ema x_attack
        #     elif i==5:
        #         input = x_attack 
        #     else:
        #         input = self.transform[i](x_ema) # x_ema x_attack
        #     J_t_median = J_t(self.model, x, input)
        #     inputs_dis.append(J_t_median) 
        # print(inputs_dis)
        
        inputs=[]
        for transform in self.transform:
            input = transform(x.clone())
            J_t_median = J_t(self.model, x, input)
            # print(J_t_median)
            inputs.append([input,J_t_median])
        # x_attack = self.attack(x,outputs) # Reduce attack
        # inputs.append([x_attack,J_t(self.model, x, x_attack)]) # Reduce attack
        J_list = sorted(inputs,key=lambda x:x[1])
        inputs = J_list[round(np.exp(epoch*0.01))-1]
        inputs_nwd = torch.cat((x, inputs[0]), dim=0)
        discrepancy_loss = -discrepancy(inputs_nwd)
        discrepancy_loss = discrepancy_loss.to(self.device)
        if inputs[1]<np.exp(epoch*0.1): # epoch * 0.1
            # print('J_t_median:',inputs[1])
            for i in range(N):
                # for t in range(len(x)):
                #     x[t]=self.transform(x[t])
                # outputs1_  = self.model_ema(self.transform1_norm(x)).detach()
                # outputs_emas.append(outputs1_*0.1)
                # outputs2_  = self.model_ema(self.transform2_sample(x)).detach()
                # # outputs_emas.append(outputs2_*0.4)
                # outputs3_  = self.model_ema(self.transform3_rot(x)).detach()
                # outputs_emas.append(outputs3_*0.4)
                # outputs4_  = self.model_ema(self.transform4_noise(x)).detach()
                # outputs_emas.append(outputs4_)
                # outputs5_  = self.model_ema(self.transform5_permu(x)).detach()
                # outputs_emas.append(outputs5_*0.1)
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
            a=0.01
            gram_loss = inputs[1]
            gram_loss = gram_loss.to(self.device)
            loss = discrepancy_loss
            loss.backward()
            optimizer.step()
            # Teacher update
            self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.rst).float().cuda(self.device) 
                        # mask=0.05
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"].cuda(self.device) * mask + p * (1.-mask)
        return standard_ema   


class CoTTA_attack_softmatch(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9, arg=None):
        super().__init__()
        self.model = model
        self.device = get_device(arg.g)
        self.model.to(self.device)
        self.optimizer = optimizer
        self.steps = steps
        self.args = arg
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
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
        
        self.T = 1.0
        self.use_hard_label = True
        self.dist_align = True
        self.ema_p = 0.999
        self.lb_prob_t = torch.ones((self.args.activity_label_size)).cuda(self.device) / self.args.activity_label_size
        self.ulb_prob_t = torch.ones((self.args.activity_label_size)).cuda(self.device) / self.args.activity_label_size
        self.prob_max_mu_t = 1.0 / self.args.activity_label_size
        self.prob_max_var_t = 1.0
        # self.amp_cm = autocast if self.args.amp else contextlib.nullcontext
        self.amp_cm = contextlib.nullcontext #config中默认false
    @torch.no_grad()
    def update_prob_t(self, lb_probs, ulb_probs):
        # if self.args.distributed and self.args.world_size > 1:
        #     lb_probs = concat_all_gather(lb_probs)
        #     ulb_probs = concat_all_gather(ulb_probs)
        
        ulb_prob_t = ulb_probs.mean(0)
        self.ulb_prob_t = self.ema_p * self.ulb_prob_t + (1 - self.ema_p) * ulb_prob_t

        lb_prob_t = lb_probs.mean(0)
        self.lb_prob_t = self.ema_p * self.lb_prob_t + (1 - self.ema_p) * lb_prob_t

        max_probs, max_idx = ulb_probs.max(dim=-1)
        prob_max_mu_t = torch.mean(max_probs)
        prob_max_var_t = torch.var(max_probs, unbiased=True)
        self.prob_max_mu_t = self.ema_p * self.prob_max_mu_t + (1 - self.ema_p) * prob_max_mu_t.item()
        self.prob_max_var_t = self.ema_p * self.prob_max_var_t + (1 - self.ema_p) * prob_max_var_t.item()
    
    @torch.no_grad()
    def calculate_mask(self, probs):
        max_probs, max_idx = probs.max(dim=-1)

        # compute weight
        mu = self.prob_max_mu_t
        var = self.prob_max_var_t
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / 4)))

        # if self.it % 250 == 0 and self.args.gpu == 0:
        #     self.print_fn(f"it {self.it}")
        #     self.print_fn(f"mean {self.prob_t}")
        #     self.print_fn(f"mu {self.prob_max_mu_t * (self.prob_t / self.prob_t.max())}")
        #     self.print_fn(f"std {math.sqrt(self.prob_max_var_t)}")
        #     pl_label_cnt = torch.bincount(max_idx, minlength=self.num_classes)
        #     w1_cnt = torch.bincount(max_idx[mask == 1.0], minlength=self.num_classes)
        #     self.print_fn(f"pl {pl_label_cnt}") 
        #     self.print_fn(f"w1 {w1_cnt}") 
        #     self.print_fn("\n")
    

        return max_probs.detach(), mask.detach()
    @torch.no_grad()
    def distribution_alignment(self, probs):
        # da
        probs = probs * self.lb_prob_t / self.ulb_prob_t
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs.detach()    
    
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
        discrepancy = NuclearWassersteinDiscrepancy(self.model)
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
            inputs.append([input,J_t_median])
        x_attack = self.attack(x,outputs)
        x_attack = x_attack.to(self.device)
        inputs.append([x_attack,J_t(self.model, x, x_attack)])
        J_list = sorted(inputs,key=lambda x:x[1])
        with self.amp_cm():
            x_ulb_w = J_list[0][0]
            x_ulb_s = J_list[5][0]
            x_ulb = torch.cat((x_ulb_w, x_ulb_s))
            logits = self.model(x_ulb)
            logits_x_ulb_w, logits_x_ulb_s = logits.chunk(2)
            probs_x_lb = torch.softmax(outputs.detach(), dim=-1)#这里本来不应该有带标签数据的，可能有问题
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

            # update 
            self.update_prob_t(probs_x_lb, probs_x_ulb_w)

            # distribution alignment
            if self.dist_align:
                probs_x_ulb_w = self.distribution_alignment(probs_x_ulb_w)

            # calculate weight
            max_probs, mask = self.calculate_mask(probs_x_ulb_w)


            # calculate loss 
            unsup_loss, _ = consistency_loss(logits_x_ulb_s,
                                             logits_x_ulb_w,
                                             'ce',
                                             use_hard_labels=self.use_hard_label,
                                             T=self.T,
                                             mask=mask)

        
        # if self.args.amp:
        #     self.scaler.scale(total_loss).backward()
        #     if (self.args.clip > 0):
        #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        #     self.scaler.step(self.optimizer)
        #     self.scaler.update()
        # else:
        #     total_loss.backward()
        #     if (self.args.clip > 0):
        #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        #     self.optimizer.step()
        
            # total_loss.backward()
            # if (self.args.clip > 0):
            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        
        inputs = J_list[round(np.exp(epoch*0.01))-1]
        inputs_nwd = torch.cat((x, inputs[0]), dim=0)
        discrepancy_loss = -discrepancy(inputs_nwd)
        discrepancy_loss = discrepancy_loss.to(self.device)
        if inputs[1]<np.exp(epoch*0.1): # epoch * 0.1
            for i in range(N):
                outputs_  = self.model_ema(inputs[0]).detach()
                "here is very important"
                outputs_emas.append(outputs_)
            if anchor_prob.mean(0)<self.ap:
                outputs_ema = torch.stack(outputs_emas).mean(0)
            else:
                outputs_ema = standard_ema
            # Student update
            optimizer.zero_grad()
            gram_loss = inputs[1]
            gram_loss = gram_loss.to(self.device)
            # print('gramloss:',gram_loss)
            # print('cross:',((softmax_entropy(outputs, outputs_ema)).mean(0)))
            # loss = (softmax_entropy(outputs, outputs_ema)).mean(0)+0.1*gram_loss
            loss = discrepancy_loss+0.01*unsup_loss
            # loss = unsup_loss
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
                            p.data = self.model_state[f"{nm}.{npp}"].cuda(self.device) * mask + p * (1.-mask)
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
def fetch_classifier(args, input=None, output=None):
    print(args)
    if 'lstm' in args.model:
        model = ClassifierLSTM(args, input=args.input, output=args.output)
    elif 'gru' in args.model:
        model = ClassifierGRU(args.model_cfg, input=args.feature_num, output=args.activity_label_size) #  args.output feature_num args.input
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
