import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import sklearn.metrics as sm
# from torchstat import stat
import torch.nn.functional as F
from torchsummary import summary
from copy import deepcopy
from torch.nn.functional import interpolate
from scipy.stats import special_ortho_group
class Preprocess4Sample():
    def __init__(self, seq_len, temporal=0.4, temporal_range=[0.8, 1.2]):
        self.seq_len = seq_len
        self.temporal = temporal
        self.temporal_range = temporal_range

    def __call__(self, instances):
        # Assuming instances have shape [batch_size, 1, time_steps, features]
        batch_size, _, time_steps, features = instances.shape
        
        if time_steps == self.seq_len:
            return instances
        
        if self.temporal > 0:
            temporal_prob = torch.rand(batch_size).to(instances.device)
            temporal_mask = temporal_prob < self.temporal
            if temporal_mask.any():
                ratio_random = torch.rand(batch_size).to(instances.device) * (self.temporal_range[1] - self.temporal_range[0]) + self.temporal_range[0]
                seq_len_scale = (ratio_random * self.seq_len).round().long()
                instances_new = torch.zeros((batch_size, 1, self.seq_len, features), device=instances.device)
                for i in range(batch_size):
                    if temporal_mask[i]:
                        # Apply interpolation to the selected instance
                        f = interpolate(instances[i:i+1, :, :, :], size=(int(seq_len_scale[i]), features), mode='linear', align_corners=False)
                        # Select a random starting point for the sequence
                        start_index = torch.randint(0, f.shape[2] - self.seq_len, (1,)).item()
                        instances_new[i, :, :, :] = f[:, :, start_index:start_index + self.seq_len, :]
                # Where temporal_mask is False, keep the original instances
                instances = torch.where(temporal_mask.view(-1, 1, 1, 1), instances_new, instances)
        
        # If not applying temporal scaling, just randomly select a sequence
        else:
            index_rand = torch.randint(0, time_steps - self.seq_len, (batch_size,))
            instances = torch.stack([instances[i, :, index_rand[i]:index_rand[i] + self.seq_len, :] for i in range(batch_size)])
        
        return instances

class Preprocess4Normalization():
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma

    def __call__(self, instance):
        # print(instance.shape)
        instance_new = instance.clone()[:, :self.feature_len]
        # if instance_new.shape[1] >= 6 and self.norm_acc:
        #     instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        if instance_new.shape[1] == 9 and self.norm_mag:
            mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
            mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
            instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        return instance_new
class Preprocess4Permute():
    
    def __init__(self, segment_size=4):
        super().__init__()
        self.segment_size = segment_size

    def __call__(self, instance):
        # Assuming instance has shape [batch_size, 1, time_steps, features]
        batch_size, _, time_steps, features = instance.shape
        # Calculate the number of segments
        num_segments = time_steps // self.segment_size
        # Reshape to have segments as the first dimension
        instance = instance.view(batch_size, num_segments, self.segment_size, features)
        # Generate a random permutation of segments
        permuted_indices = torch.randperm(num_segments)
        # Apply the permutation to the segments
        instance = instance[:, permuted_indices, :, :]
        # Reshape back to the original shape
        instance = instance.view(batch_size, 1, time_steps, features)
        return instance


class Preprocess4Rotation():
    def __init__(self, sensor_dimen=3):
        super().__init__()
        self.sensor_dimen = sensor_dimen

    def __call__(self, instance):
        return self.rotate_random(instance)

    def rotate_random(self, instance):
        # Assuming instance has shape [batch_size, 1, time_steps, features]
        batch_size, _, time_steps, features = instance.shape
        instance_new = instance.view(batch_size, time_steps, features // self.sensor_dimen, self.sensor_dimen)
        rotation_matrix = torch.tensor(special_ortho_group.rvs(self.sensor_dimen)).float().to(instance.device)
        instance_new = torch.einsum('bijk,kl->bijl', instance_new, rotation_matrix)
        return instance_new.view(batch_size, 1, time_steps, features)
class Preprocess4Noise():
    def __init__(self, p=1.0, mu=0.0, var=0.1):
        super().__init__()
        self.p = p
        self.mu = mu
        self.var = var

    def __call__(self, instance):
        if torch.rand(1).item() < self.p:
            noise = torch.normal(mean=self.mu, std=self.var, size=instance.shape).to(instance.device)
            instance += noise
        return instance

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model
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
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        # self.transform = [Preprocess4Normalization(arg.input),Preprocess4Sample(arg.len_sw, temporal=0.4)
        #     , Preprocess4Rotation(), Preprocess4Noise(), Preprocess4Permute()]
        self.transform = [Preprocess4Normalization(arg.n_class),Preprocess4Sample(arg.len_sw, temporal=0.4)
                          ,Preprocess4Rotation(), Preprocess4Noise(), Preprocess4Permute()]
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

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
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        print(x.shape)
        # print(x[0].shape)
        # print(len(x))
        outputs = self.model(x)
        # print('output:',outputs.shape)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        # Teacher Prediction
        s1,_ = self.model_anchor(x)
        anchor_prob = torch.nn.functional.softmax(s1, dim=1).max(1)[0]
        standard_ema,_ = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = 32 
        outputs_emas = []
        for i in range(N):
            # for t in range(len(x)):
            #     x[t]=self.transform(x[t])
            # outputs_  = self.model_ema(self.transform(x)).detach()
            "here is very important"
            for transform in self.transform:
                x = transform(x)
            s2,_ = self.model_ema(x)
            outputs_  = s2.detach()
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




class CNN_UCI(nn.Module):
    def __init__(self):
        super(CNN_UCI, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
            )
        self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
        )
        self.layer5 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
        self.layer6 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
        )
        self.classifier = nn.Linear(15360, 6)


    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        feature = x
        x = self.classifier(x)
        # x = nn.LayerNorm(x.size())(x.cpu())
        # x = x.cuda()
        # x = F.normalize(x.cuda())
        return x, feature

class ResCNN_UCI(nn.Module):
    def __init__(self):
            super(ResCNN_UCI, self).__init__()
            self.Block1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
            self.shortcut1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(64),
            )
            self.Block2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            self.shortcut2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(128),
            )
            self.Block3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(256),
                nn.Dropout(0.5),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                # nn.Dropout(0.5),
                nn.ReLU(True)
            )
            self.shortcut3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(256),
                # nn.Dropout(0.5)
            )
            self.classifier = nn.Linear(15360, 6)


    def forward(self, x):
        h1 = self.Block1(x)
        r = self.shortcut1(x)
        h1 = h1 + r
        h2 = self.Block2(h1)
        r = self.shortcut2(h1)
        h2 = h2 + r
        h3 = self.Block3(h2)
        r = self.shortcut3(h2)
        h3 = h3 + r
        x = h3.view(h3.size(0), -1)
        feature = x
        x = self.classifier(x)
        # x = nn.LayerNorm(x.size())(x.cpu())
        # x = x.cuda()
        x = F.normalize(x.cuda())
        return x, feature


class CNN_UNIMIB(nn.Module):
    def __init__(self):
            super(CNN_UNIMIB, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
            self.layer5 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
            )
            self.layer6 = nn.Sequential(
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
            )
            self.classifier = nn.Linear(8448, 17)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        feature = x
        x = self.classifier(x)

        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        # x = F.normalize(x.cuda())
        return x,feature


class ResCNN_UNIMIB(nn.Module):
    def __init__(self):
            super(ResCNN_UNIMIB, self).__init__()
            self.Block1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            self.shortcut1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
            )
            self.Block2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )
            self.shortcut2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
            )
            self.Block3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True)
            )
            self.shortcut3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
            )
            self.classifier = nn.Linear(8448, 17)


    def forward(self, x):
        h1 = self.Block1(x)
        r = self.shortcut1(x)
        h1 = h1 + r
        h2 = self.Block2(h1)
        r = self.shortcut2(h1)
        h2 = h2 + r
        h3 = self.Block3(h2)
        r = self.shortcut3(h2)
        h3 = h3 + r
        x = h3.view(h3.size(0), -1)
        feature = x
        x = self.classifier(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        # x = F.normalize(x.cuda())
        return x, feature

" OPPORTUNITY "
class CNN_OPPORTUNITY(nn.Module):
    def __init__(self,  num_class=17):
        super(CNN_OPPORTUNITY, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.classifier = nn.Linear(1024, num_class)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
    
        out = out.view(out.size(0), -1)
        feature = out
        # print(out.shape)
        out = self.classifier(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out,feature

class ResCNN_OPPORTUNITY(nn.Module):
    def __init__(self, num_class=17):
        super(ResCNN_OPPORTUNITY, self).__init__()

        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2)),
            nn.BatchNorm2d(64),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2)),
            nn.BatchNorm2d(128),
        )
        
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2)),
            nn.BatchNorm2d(512),
        )

        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.classifier = nn.Linear(1024, num_class)


    def forward(self, x):

        h1 = self.Block1(x)
        r = self.shortcut1(x)
        h1 = h1 + r
        h1 = self.pool1(h1)

        h2 = self.Block2(h1)
        r = self.shortcut2(h1)
        h2 = h2 + r
        h2 = self.pool2(h2)


        h3 = self.Block3(h2)
        r = self.shortcut3(h2)
        h3 = h3 + r
        out = self.pool3(h3)
    
        out = out.view(out.size(0), -1)
        # print(out.shape)
        feature = out
        out = self.classifier(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out, feature


def CNN_choose(dataset = 'uci', res=False):
    if dataset == 'uci':
        
        if res == False:
            # print('1')
            model = CNN_UCI()
        else:
            model = ResCNN_UCI()
        return model

    if dataset == 'unimib':
        if res == False:
            model = CNN_UNIMIB()
        else:
            model = ResCNN_UNIMIB()
        return model

    if dataset == 'oppo':
        if res == False:
            model = CNN_OPPORTUNITY()
        else:
            model = ResCNN_OPPORTUNITY()
        return model
    else:
        return print('not exist this model')

        
@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

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
        



def main():
    model = CNN_choose(dataset = 'unimib', res=False).cuda()
    input = torch.rand(1, 1, 151, 3).cuda()
    output = model(input)
    print(output.shape)
    summary(model, (1, 151, 3))
    
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))

if __name__ == '__main__':
    main()