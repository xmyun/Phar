import torch
from torch import nn as nn
import torch.nn.functional as F


class AverageMeter(object):
    """
    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


def consistency_loss(logits_s, logits_w, name='ce', use_hard_labels=True, T=0.5, mask=None, label_smoothing=0.0, softmax=True):
    assert name in ['ce', 'mse']
    logits_w = logits_w.detach()
    if name == 'mse':
        if softmax:
            pseudo_label = torch.softmax(logits_w, dim=-1)
        else:
            pseudo_label = logits_w
        probs = torch.softmax(logits_s, dim=-1)
        loss = F.mse_loss(probs, pseudo_label, reduction='none').sum(dim=1)
    else:
        if use_hard_labels:
            pseudo_label = torch.argmax(logits_w, dim=-1)
            if label_smoothing:
                pseudo_label = smooth_targets(logits_w, pseudo_label, label_smoothing)
                use_hard_labels = False
            loss = ce_loss(logits_s, pseudo_label, use_hard_labels, reduction='none')
        else:
            if softmax:
                pseudo_label = torch.softmax(logits_w / T, dim=-1)
            else:
                pseudo_label = logits_w
            loss = ce_loss(logits_s, pseudo_label, use_hard_labels, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean(), pseudo_label


class Bn_Controller:
    def __init__(self):
        """
        freeze_bn and unfreeze_bn must appear in pairs
        """
        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                self.backup[name + '.running_mean'] = m.running_mean.data.clone()
                self.backup[name + '.running_var'] = m.running_var.data.clone()
                self.backup[name + '.num_batches_tracked'] = m.num_batches_tracked.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + '.running_mean']
                m.running_var.data = self.backup[name + '.running_var']
                m.num_batches_tracked.data = self.backup[name + '.num_batches_tracked']
        self.backup = {}


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


class EMA:
    """
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def smooth_targets(logits, targets, smoothing=0.1):
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (logits.shape[-1] - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), (1 - smoothing))
    return true_dist


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output