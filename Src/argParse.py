
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

# from config import create_io_config, load_dataset_config, TrainConfig, MaskConfig, load_model_config


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def stat_acc_f1(label, results_estimated):
    # label = np.concatenate(label, 0)
    # results_estimated = np.concatenate(results_estimated, 0)
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
    # return np.arange(user_domain_mapping.shape[0])[index].astype(np.int)

# handle_argv(target, config_train)
def set_arg(): 
    parser = argparse.ArgumentParser(description='Uhar tta_imu')
    parser.add_argument('--model', type=str, help='The model you want to useodel.')
    parser.add_argument('--dataset', type=str, help='Dataset name.', choices=['hhar_20_120', 'motion_20_120', 'uci_20_120', 'shoaib_20_120'])
    parser.add_argument('--source_dataset', type=str, default="uci_20_120", help='source model name', choices=['hhar_20_120', 'motion_20_120', 'uci_20_120', 'shoaib_20_120'])
    # parser.add_argument('dataset_version', type=str, help='Dataset version')
    # parser.add_argument('-d', '--domain', type=int, default=0, help='The domain index.')
    # parser.add_argument('-td', '--target_domain', type=int, default=0, help='The target domain index')
    # parser.add_argument('-e', '--encoder', type=str, default=None, help='Pretrain encoder and decoder file.')
    # parser.add_argument('-c', '--classifier', type=str, default=None, help='Trained classifier file.')
    # parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU.')
    # parser.add_argument('-t', '--cfg_train', type=str, default='./config/' + config_train, help='Training config json file path')
    # parser.add_argument('-m', '--cfg_mask', type=str, default='./config/mask.json', help='Mask strategy json file path')
    # parser.add_argument('-l', '--label_index', type=int, default=0, help='Label Index')
    # parser.add_argument('-n', '--label_number', type=int, default=1000, help='Label number')
    # parser.add_argument('-rda', '--remove_data_augmentation', type=int, default=0, help='Specify whether adopting data augmentations.')
    # parser.add_argument('-s', '--save_name', type=str, default='model', help='The saved model name')
    # parser.add_argument('-pt', '--print_time', type=int, default=0, help='Label number')
    # parser.add_argument('--eval', action='store_true', help='Specify evaluation mode.')
    args = parser.parse_args()
    json_data_path = '/home/xuemeng/IMU/IMU_Hete/config/dataset.json'
    json_model_path = '/home/xuemeng/IMU/IMU_Hete/config/model.json'
    json_train_path = '/home/xuemeng/IMU/IMU_Hete/config/train.json'
    json_mask_path = '/home/xuemeng/IMU/IMU_Hete/config/mask.json'
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


# def load_classifier_data_config(args):
#     model_cfg = args.model_cfg
#     train_cfg = TrainConfig.from_json(args.train_cfg)
#     dataset_cfg = args.dataset_cfg
#     set_seeds(train_cfg.seed)
#     data = np.load(args.data_path).astype(np.float32)
#     labels = np.load(args.label_path).astype(np.float32)
#     return data, labels, train_cfg, model_cfg, dataset_cfg


# def load_classifier_config(args):
#     model_cfg = args.model_cfg
#     train_cfg = TrainConfig.from_json(args.train_cfg)
#     dataset_cfg = args.dataset_cfg
#     set_seeds(train_cfg.seed)
#     return train_cfg, model_cfg, dataset_cfg


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
            # print(print_info[-1])
            # print(message)

