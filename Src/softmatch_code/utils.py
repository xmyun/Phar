import math
import os
import time

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import logging
import yaml

from custom_writer import CustomWriter


def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def setattr_cls_from_kwargs(cls, kwargs):
    # if default values are in the cls,
    # overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls, key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])


def net_builder(net_name, from_name: bool, net_conf=None, is_remix=False):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
    """
    if from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(models.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return models.__dict__[net_name]

    else:
        if net_name == 'WideResNet':
            import torchssl.nets.wrn as net
            builder = getattr(net, 'build_WideResNet')()
        elif net_name == 'WideResNetVar':
            import torchssl.nets.wrn_var as net
            builder = getattr(net, 'build_WideResNetVar')()
        elif net_name == 'ResNet50':
            import torchssl.nets.resnet50 as net
            builder = getattr(net, 'build_ResNet50')(is_remix)
        else:
            assert Exception("Not Implemented Error")

        if net_name != 'ResNet50':
            setattr_cls_from_kwargs(builder, net_conf)
        return builder.build


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """

    def __init__(self, tb_dir, file_name, use_tensorboard=False):
        self.tb_dir = tb_dir
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))
        else:
            self.writer = CustomWriter(os.path.join(self.tb_dir, file_name))

    def update(self, tb_dict, it, suffix=None, mode="train"):
        """
        Args
            tb_dict: contains scalar values for updating tensorboard
            it: contains information of iteration (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ''
        if self.use_tensorboard:
            for key, value in tb_dict.items():
                self.writer.add_scalar(suffix + key, value, it)
        else:
            self.writer.set_epoch(it, mode)
            for key, value in tb_dict.items():
                self.writer.add_scalar(suffix + key, value)
            self.writer.plot_stats()
            self.writer.dump_stats()


def get_optimizer(net, optim_name='SGD', lr=0.1, momentum=0.9, weight_decay=0, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''

    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name or 'bias' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]

    if optim_name == 'SGD':
        optimizer = torch.optim.SGD(per_param_args, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                    nesterov=nesterov)
    elif optim_name == 'AdamW':
        optimizer = torch.optim.AdamW(per_param_args, lr=lr, weight_decay=weight_decay)
    return optimizer


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''

    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)