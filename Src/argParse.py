
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import random
import json
import numpy as np
import torch
from scipy import signal, interpolate
from scipy.stats import special_ortho_group
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from typing import NamedTuple
import sys


def set_seeds(seed):
    "set random seeds"
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    
    # # New add, Trade speed to fix the program's results. 
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.backends.cudnn.deterministic = True

def stat_acc_f1(label, results_estimated):
    label_estimated = np.argmax(results_estimated, 1)
    f1 = f1_score(label, label_estimated, average='macro')
    acc = np.sum(label == label_estimated) / label.size
    return acc, f1

def filter_domain_user_list(labels, source_domain, target_domain, label_user_index=1, label_domain_index=2, domain_num=4):
    if source_domain == 0:
        domain_other = [i + 1 for i in range(domain_num)]
    elif target_domain == 0:
        domain_other = [i + 1 for i in range(domain_num)]
        domain_other.remove(source_domain)
    else:
        domain_other = [target_domain]
    user_domain_mapping = np.unique(labels[:, 0, [label_user_index, label_domain_index]], axis=0)
    index = np.isin(user_domain_mapping[:, 1], np.array(domain_other) - 1)
    return user_domain_mapping[index, 0].astype(np.int)

def set_arg(): 
    parser = argparse.ArgumentParser(description='Phar tta_imu') 
    parser.add_argument('--dataset', type=str, help='Dataset name.')
    parser.add_argument('--model', type=str, help='The model you want to useodel.')
    parser.add_argument('--cd', type=str, default="SigDom", help='Whether the source damain and target domain in a same dataset.')
    # parser.add_argument('--dataset', type=str, help='Dataset name.', choices=['hhar_20_120', 'motion_20_120', 'uci_20_120', 'shoaib_20_120'])
    parser.add_argument('--source_dataset', type=str, default="Dataset without target!", help='source model name', choices=['hhar_20_120', 'motion_20_120', 'uci_20_120', 'shoaib_20_120', 'new_20_120', 'Dataset without target!'])
    parser.add_argument('--SDom', type=str, help='Source domain.', choices=['aet0', 'hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6'])
    parser.add_argument('--TDom', type=str, help='Target domain.', choices=['hhar1', 'uci2', 'motion3', 'shoaib4', 'usc5', 'ku6'])
    parser.add_argument('--g', type=str, default=None, help='Set specific GPU.')
    
    args = parser.parse_args()
    args = load_model_config(args, args.model)
    
    
    json_data_path = '/mnt/home/xuemeng/ttaIMU/IMU_Hete/config/dataset.json'
    json_model_path = '/mnt/home/xuemeng/ttaIMU/IMU_Hete/config/model.json'
    json_train_path = '/mnt/home/xuemeng/ttaIMU/IMU_Hete/config/train.json'
    json_mask_path = '/mnt/home/xuemeng/ttaIMU/IMU_Hete/config/mask.json'
    with open(json_model_path) as f:
        json_data = json.load(f)
        model_config = json_data.get(args.model)
    with open(json_data_path) as f:
        json_data = json.load(f)
        data_config = json_data.get(args.dataset)
    with open(json_train_path) as f:
        train_config = json.load(f)
    with open(json_mask_path) as f:
        mask_config = json.load(f)
    args.__dict__.update(train_config)
    args.__dict__.update(data_config)
    args.__dict__.update(model_config)
    args.__dict__.update(mask_config)
    return args

def manual_models_path(args, target, models):
    save_path_pretrain = os.path.join('saved', target + "_" + args.dataset + "_" + args.dataset_version)
    models_new = [os.path.join(save_path_pretrain, model) for model in models]
    return models_new


def read_dataset(path, version):
    path_data = os.path.join(path, 'data_' + version + '.npy')
    path_label = os.path.join(path, 'label_' + version + '.npy')
    return np.load(path_data), np.load(path_label)


def load_data(args):
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class PretrainModelConfig(NamedTuple):

    feature_num: int = 0  #
    hidden: int = 0  # Dimension of Hidden Layer in Transformer Encoder
    hidden_ff: int = 0  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    n_layers: int = 0  # Numher of Hidden Layers
    n_heads: int = 0  # Numher of Heads in Multi-Headed Attention Layers
    seq_len: int = 0  # Maximum Length for Positional Embeddings
    emb_norm: bool = False

    @classmethod
    def from_json(cls, js):
        return cls(**js)

class ClassifierModelConfig(NamedTuple):
    "Configuration for classifier model"
    seq_len: int = 0
    feature_num: int = 0

    num_rnn: int = 0
    num_layers: int = 0
    rnn_io: list = []
    rnn_bidirection: list = []

    num_cnn: int = 0
    conv_io: list = []
    pool: list = []
    flat_num: int = 0

    num_attn: int = 0
    num_head: int = 0
    atten_hidden: int = 0

    num_linear: int = 0
    linear_io: list = []

    activ: bool = False
    dropout: bool = False

    encoder: str = None
    encoder_cfg: object = PretrainModelConfig
    
    input: int= 6
    save_path: str=None


    @classmethod
    def from_json(cls, js):
        return cls(**js)

    @classmethod
    def from_encoder_cfg(cls, js, encoder_cfg):

        cfg = cls(**js)
        cfg.encoder_cfg = encoder_cfg
        return cfg


def load_model_config(args, model, path_bert='/mnt/home/xuemeng/ttaIMU/IMU_Hete/config/encoder.json', path_classifier='/mnt/home/xuemeng/ttaIMU/IMU_Hete/config/model.json'):
    model_cfg = None
    model_config_all = json.load(open(path_classifier, "r"))
    if model in model_config_all:
        model_cfg = ClassifierModelConfig.from_json(model_config_all[model])
        if model_cfg.encoder is not None:
            model_encoder_config_all = json.load(open(path_bert, "r"))
            if model_cfg.encoder in model_encoder_config_all:
                encoder_cfg = PretrainModelConfig.from_json(model_encoder_config_all[model_cfg.encoder])
                args.encoder_cfg = encoder_cfg
            else:
                print("Unable to find model encoder config!")
    if model_cfg is None:
        print("Unable to find model config!")
        sys.exit()
    args.model_cfg = model_cfg
    return args


def print_stat(stats, labels, source_domain, domain_size):
    print_info = ['All domain users', 'Other domain users', 'Cross domain users']
    message = ''
    for i in range(domain_size + 2):
        sd = 0 if i == 0 else source_domain
        td = i - 1 if i > 1 else 0
        user_index = filter_domain_user_list(labels, sd, td, domain_num=domain_size)
        mean_acc_f1 = np.mean(stats[user_index, :], axis=0)
        std_acc_f1 = np.std(stats[user_index, :], axis=0)
        if i > 1:
            message += '[%0.3f, %0.3f; %0.3f, %0.3f]' % (mean_acc_f1[0], std_acc_f1[0], mean_acc_f1[1], std_acc_f1[1])
        else:
            print(print_info[i])
            print('[Accuracy; F1]: [%0.3f, %0.3f; %0.3f, %0.3f]' % (mean_acc_f1[0], std_acc_f1[0], mean_acc_f1[1], std_acc_f1[1]))
        if i == domain_size + 1:
            pass
