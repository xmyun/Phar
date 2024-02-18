# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import math
import os
import random

import numpy as np
import torch

from scipy import signal, interpolate
from scipy.stats import special_ortho_group
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from dataAug import IMUDataset, IMUTTADataset, FFTDataset
from dataAug import Preprocess4Normalization, Preprocess4Sample, Preprocess4Rotation, Preprocess4Noise, Preprocess4Permute
from argParse import set_seeds

# from config import create_io_config, load_dataset_config, TrainConfig, MaskConfig, load_model_config


def get_device(gpu, print_info=True):
    "get device (CPU or GPU)"
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if print_info:
        print("%s (%d GPUs)" % (device, n_gpu))
    return device


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def filter_dataset(data, label, filter_index=[0, 1]):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        temp_label = np.unique(label[i, :, filter_index], axis=1)
        # if temp_label.shape == (2, 1): # This will filter out some datas. 
        index[i] = True
        label_new.append(label[i, 0, :])
    # print('Before Merge: %d, After Merge: %d' % (data.shape[0], np.sum(index)))
    return data[index], np.array(label_new)


def reshape_data(data_set, seq_len):
    if seq_len == 0:
        return data_set
    result = []
    for item in data_set:
        result.append(item.reshape(item.shape[0] * item.shape[1] // seq_len, seq_len, item.shape[2]))
    return result


def reshape_label(label_set, seq_len_original, seq_len):
    if seq_len == 0:
        return label_set
    result = []
    for item in label_set:
        item = np.repeat(item, seq_len_original // seq_len, axis=0)
        result.append(item)
    return result


def shuffle_data_label(data, label):
    index = np.random.permutation(np.arange(data.shape[0]))
    return data[index, ...], label[index, ...]


def select_random_data_label(data, label, sample_num):
    index = np.random.choice(label.shape[0], sample_num, replace=False)
    return data[index, ...], label[index, ...]


def filter_domain(data, labels, domain, label_domain_index=2):
    index = labels[:, 0, label_domain_index] == (domain - 1)
    return data[index, ...], labels[index, ...]


def filter_domain_list(data, labels, domain_list, label_domain_index=2):
    if (isinstance(domain_list, int) and domain_list == 0) or (isinstance(domain_list, list) and domain_list == [0]):
        return data, labels
    else:
        index = np.isin(labels[:, 0, label_domain_index], np.array(domain_list) - 1)
        return data[index, ...], labels[index, ...]



def filter_user(data, labels, labels_user, user):
    index = labels_user == user
    return data[index, ...], labels[index, ...]


def filter_label(data, labels, label_target):
    index = labels == label_target
    return data[index, ...], labels[index, ...]


def filter_label_max(data, labels, label_max, label_index=0):
    label_target = np.arange(label_max)
    index = np.isin(labels[:, label_index], label_target)
    return data[index, ...], labels[index, ...]


def filter_act_user_domain(data, labels, act=None, user=None, domain=None, label_act_max=None):
    index = np.ones(labels.shape[0]).astype(np.bool_)
    if act is not None:
        index = index & (labels[:, 0, 0] == act)
    if label_act_max is not None:
        index = index & (labels[:, 0, 0] < label_act_max)
    if user is not None:
        index = index & (labels[:, 0, 1] == user)
    if domain is not None:
        index = index & (labels[:, 0, 2] == (domain - 1))
    return data[index, ...], labels[index, ...]


def check_domain_user_num(labels, domain, label_user_index=1, label_domain_index=2, domain_num=4):
    if domain == 0:
        domain_list = [i + 1 for i in range(domain_num)]
    else:
        domain_list = [domain]
    index = np.isin(labels[:, 0, label_domain_index], np.array(domain_list) - 1)
    return np.unique(labels[index, 0, label_user_index]).size


def separate_user(data, labels, user_label_index=1):
    users = np.unique(labels[:, user_label_index])
    result = []
    for i in range(users.size):
        result.append(filter_user(data, labels, labels[:, user_label_index], users[i]))
    return result


def separate_dataset(data, labels, domain, label_index, label_max=None):
    if domain is not None:
        # data, labels = filter_domain(data, labels, domain)
        data, labels = filter_domain_list(data, labels, domain)
    data, labels = filter_dataset(data, labels)
    if label_max is not None and label_max > 0:
        data, labels = filter_label_max(data, labels, label_max)
    dataset = separate_user(data, labels)
    data_set = [item[0] for item in dataset]
    label_set = [item[1][:, label_index] for item in dataset]
    return data_set, label_set


def prepare_dataset(data, labels, seq_len, training_size=0.8, test_size=0.1, domain=0, label_index=0, merge=False, label_max=None, seed=None):
    data_set, label_set = separate_dataset(data, labels, domain, label_index, label_max=label_max)
    result = [[], [], [], [], [], []]
    for i in range(len(data_set)):
        data_train_temp, data_test, label_train_temp, label_test \
            = train_test_split(data_set[i], label_set[i], train_size=training_size, random_state=seed)
        training_size_new = training_size / (1 - test_size)
        data_train, data_vali, label_train, label_vali \
            = train_test_split(data_train_temp, label_train_temp, train_size=training_size_new, random_state=seed)

        [data_train, data_vali, data_test] = reshape_data([data_train, data_vali, data_test], seq_len)
        [label_train, label_vali, label_test] = reshape_label([label_train, label_vali, label_test],
                                                                    data.shape[1], seq_len)
        result[0].append(data_train)
        result[1].append(label_train)
        result[2].append(data_vali)
        result[3].append(label_vali)
        result[4].append(data_test)
        result[5].append(label_test)
    if merge:
        for i in range(len(result)):
            if i % 2 == 0:
                result[i] = np.vstack(result[i])
            else:
                if isinstance(label_index, list):
                    result[i] = np.vstack(result[i])
                else:
                    result[i] = np.concatenate(result[i])
    return result


def prepare_classification_dataset(data, labels, seq_len, label_index, training_size, domain=None, label_max=None, seed=None):
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = prepare_dataset(data, labels, seq_len=0, domain=domain, label_index=label_index, merge=True
                          , label_max=label_max, seed=seed)
    if training_size == 1.0:
        data_train_label, label_train_label = data_train, label_train
    else:
        data_train_label, _, label_train_label, _ = train_test_split(data_train, label_train, train_size=training_size, random_state=seed)
    [data_train_label, data_vali, data_test] = reshape_data([data_train_label, data_vali, data_test], seq_len)
    [label_train_label, label_vali, label_test] = reshape_label([label_train_label, label_vali, label_test], data.shape[1],
                                                          seq_len)
    return data_train_label, label_train_label, data_vali, label_vali, data_test, label_test


def prepare_classification_da_dataset(data, labels, seq_len, label_index, training_size, source_domain=None, target_domain=None
                                      , label_max=None, seed=None, domain_num=4):
    data_train_label, label_train_label, data_vali, label_vali, data_test, label_test \
        = prepare_classification_dataset(data, labels, seq_len, label_index, training_size
                                         , domain=source_domain, label_max=label_max, seed=seed)
    if target_domain == 0:
        domain_other = [i + 1 for i in range(domain_num)]
        domain_other.remove(source_domain)
    else:
        domain_other = [target_domain]
    data_train_target, label_train_target, _, _, _, _ \
        = prepare_classification_dataset(data, labels, seq_len, label_index, 1.0
                                         , domain=domain_other, label_max=label_max, seed=seed)
    return data_train_label, label_train_label, data_train_target, label_train_target, data_vali, label_vali, data_test, label_test


def select_by_metric(label, select_num, metric=None, random_rate=0.0):
    select_index = np.zeros(label.size, dtype=np.bool)
    label_count = [0] * np.unique(label).size
    if random_rate < 1.0:
        loss_sort_index = metric.argsort()
        top_k = int(select_num * (1 - random_rate))
        for i in range(metric.size):
            if label_count[int(label[loss_sort_index[i]])] < top_k:
                select_index[loss_sort_index[i]] = True
                label_count[int(label[loss_sort_index[i]])] += 1
                if label_count == [top_k] * len(label_count):
                    break
    if random_rate > 0.0:
        index_random = np.random.permutation(label.size)
        for i in range(label.size):
            if label_count[int(label[index_random[i]])] < select_num and not select_index[index_random[i]]:
                select_index[index_random[i]] = True
                label_count[int(label[index_random[i]])] += 1
                if label_count == [select_num] * len(label_count):
                    break
    return select_index



#############
# Dataset processing and load points:
#############
def merge_dataset(data, label, mode='all'):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        if mode == 'all':
            temp_label = np.unique(label[i])
            if temp_label.size == 1:
                index[i] = True
                label_new.append(label[i, 0])
        elif mode == 'any':
            index[i] = True
            if np.any(label[i] > 0):
                temp_label = np.unique(label[i])
                if temp_label.size == 1:
                    label_new.append(temp_label[0])
                else:
                    label_new.append(temp_label[1])
            else:
                label_new.append(0)
        else:
            index[i] = ~index[i]
            label_new.append(label[i, 0])
    # print('Before Merge: %d, After Merge: %d' % (data.shape[0], np.sum(index)))
    return data[index], np.array(label_new)

def prepare_simple_dataset(data, labels, training_rate=0.2):
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    data_train = data[:train_num, ...]
    data_test = data[train_num:, ...]
    t = np.min(labels)
    label_train = labels[:train_num] - t
    label_test = labels[train_num:] - t
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    print('Label Size: %d, Unlabel Size: %d. Label Distribution: %s'
          % (label_train.shape[0], label_test.shape[0], ', '.join(str(e) for e in label_num)))
    return data_train, label_train, data_test, label_test

def partition_and_reshape(data, labels, label_index=0, training_rate=0.8, vali_rate=0.1, change_shape=True
                          , merge=20, merge_mode='all', shuffle=False):
    arr = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    vali_num = int(data.shape[0] * vali_rate)
    data_train = data[:train_num, ...]
    data_vali = data[train_num:train_num+vali_num, ...]
    data_test = data[train_num+vali_num:, ...]
    t = np.min(labels[:, :, label_index])
    label_train = labels[:train_num, ..., label_index] - t
    label_vali = labels[train_num:train_num+vali_num, ..., label_index] - t
    label_test = labels[train_num+vali_num:, ..., label_index] - t
    if change_shape:
        data_train = reshape_data(data_train, merge)
        data_vali = reshape_data(data_vali, merge)
        data_test = reshape_data(data_test, merge)
        label_train = reshape_label(label_train, merge)
        label_vali = reshape_label(label_vali, merge)
        label_test = reshape_label(label_test, merge)
    if change_shape and merge != 0:
        data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
        data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)
        data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
    print('Train Size: %d, Vali Size: %d, Test Size: %d' % (label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
    return data_train, label_train, data_vali, label_vali, data_test, label_test

# 读取每个用户的数据以及相应的标签
def Extractor_Users(data, labels, Tarsers, user_label_index=1):
    users = np.unique(labels[:, user_label_index])
    result = []
    print(users)
    print(data.shape)
    print(labels.shape)
    print(users.shape)
    print("User list wanted to extract", Tarsers)
    for i in Tarsers:
        if np.isin(i, users): 
            print("Extracted user:",i)
            Uindex_tmp = np.where(users == i)
            print(users[Uindex_tmp], Uindex_tmp)
            result.append(filter_user(data, labels, labels[:, user_label_index], users[Uindex_tmp]))
    return result

# 读取 指定的 用户: User_list
def load_Uci_users(User_list, data, label):
    datas, labels = filter_domain_list(data, label, 1)  # Extract data in dataset/domain 1; 1
    datas, labels = filter_dataset(datas, labels)
    datas, labels = filter_label_max(datas, labels, 4) # Cutoff four activity. 
    dataset = Extractor_Users(datas, labels, User_list, user_label_index=1)
    
    data_set = [item[0] for item in dataset]
    label_set = [item[1][:, 0] for item in dataset]
    return data_set, label_set

# 读取 指定的 用户: User_list
def load_Across_users(User_list, data, label):
    datas, labels = filter_domain_list(data, label, 0)  # Extract data in dataset/domain 1;
    datas, labels = filter_dataset(datas, labels)
    datas, labels = filter_label_max(datas, labels, 4) # Cutoff four activity. 
    dataset = Extractor_Users(datas, labels, User_list, user_label_index=1)
    
    data_set = [item[0] for item in dataset]
    label_set = [item[1][:, 0] for item in dataset]
    return data_set, label_set

def User_select(totalUser):
    B = np.array([i for i in range(0, totalUser)])
    test_ratio = 0.1
    val_ratio= 0.1
    # 创建一个包含B中所有索引的数组
    indices = np.arange(len(B))
    # 从indices中随机选择10%的索引
    test_num = min(math.ceil(len(B) * 0.1), len(indices))
    test_indices = np.random.choice(indices, size=test_num, replace=False)
    # 从indices中删除已经被选择的索引
    remaining_indices = np.delete(indices, test_indices)
    # 从remaining_indices中随机选择10%的索引
    val_num = min(math.ceil(len(B) * 0.1), len(remaining_indices))
    val_indices = np.random.choice(remaining_indices, size=val_num, replace=False)
    # 找到val_indices中的值在remaining_indices中的位置
    val_indices_positions = np.where(np.isin(remaining_indices, val_indices))[0]
    # 删除这些位置
    train_indices = np.delete(remaining_indices, val_indices_positions)
    # 使用选择的索引从B中取出对应的值
    test_values = B[test_indices]
    val_values = B[val_indices]
    train_values = B[train_indices]
    print("Test Users:", test_values, len(test_values))
    print("Validation Users:", val_values, len(val_values))
    print("Train Users:", train_values,len(train_values))
    return test_values, val_values, train_values

# 将用户分割 为 Train/valid/test
def separate_user_tvt(data, data_set, label_set, seed=None):
    print("selected users", len(label_set))
    test_G, val_G, train_G = User_select(len(label_set))
    train_size_m = 0.7 
    test_size_m = 0.1 
    result = [[], [], [], [], [], []] 
    # For each user, confiure train/vali/test. 
    for i in range(len(data_set)): 
        data_train_temp, data_test, label_train_temp, label_test \
             = train_test_split(data_set[i], label_set[i], train_size=train_size_m, random_state=seed) 
        training_size_new = train_size_m / (1 - test_size_m) 
        print("第",i,"个人中，多少比例的数据用于训练/验证/测试") 
        data_train, data_vali, label_train, label_vali \
            = train_test_split(data_train_temp, label_train_temp, train_size=training_size_new, random_state=seed) 
        [data_train, data_vali, data_test] = reshape_data([data_train, data_vali, data_test], 0)
        [label_train, label_vali, label_test] = reshape_label([label_train, label_vali, label_test],
                                                                    data.shape[1], 0)
        # 针对不同人进行不同挂载，而不是直接挂载；
        # 不同需求，不同挂载
        if i in test_G:
            print("Test Users:", i)
            result[4].append(data_train)
            result[5].append(label_train)
            result[4].append(data_vali)
            result[5].append(label_vali)
            result[4].append(data_test)
            result[5].append(label_test)
        elif i in val_G:
            print("Validation Users:", i)
            result[2].append(data_train)
            result[3].append(label_train)
            result[2].append(data_vali)
            result[3].append(label_vali)
            result[2].append(data_test)
            result[3].append(label_test)
        elif i in train_G:
            print("Train Users:", i)
            result[0].append(data_train)
            result[1].append(label_train)
            result[0].append(data_vali)
            result[1].append(label_vali)
            result[0].append(data_test)
            result[1].append(label_test)
        else:
            print("Data split wrong!!!")
            result[0].append(data_train)
            result[1].append(label_train)
            result[2].append(data_vali)
            result[3].append(label_vali)
            result[4].append(data_test)
            result[5].append(label_test)
        
    for i in range(len(result)):
        if i % 2 == 0:
            result[i] = np.vstack(result[i])
        else:
            if isinstance(0, list):
                result[i] = np.vstack(result[i])
            else:
                result[i] = np.concatenate(result[i])
    return result

def further_split_train(data, splitTrain_again, result_tvt, seq_len=0):
    data_train, label_train, data_vali, label_vali, data_test, label_test = result_tvt
    training_size = splitTrain_again
    if training_size == 1.0:
        data_train_label, label_train_label = data_train, label_train
    else:
        data_train_label, _, label_train_label, _ = train_test_split(data_train, label_train, train_size=training_size, random_state=seed)
    [data_train_label, data_vali, data_test] = reshape_data([data_train_label, data_vali, data_test], seq_len)
    [label_train_label, label_vali, label_test] = reshape_label([label_train_label, label_vali, label_test], data.shape[1],
                                                          seq_len)
    return data_train_label, label_train_label, data_vali, label_vali, data_test, label_test

def load_dataset(args):
    # print(args)
    set_seeds(args.seed)
    print("种子的数值")
    path_label = os.path.join("/label_" + "20_120" + ".npy") 
    path_data = os.path.join("/data_" + "20_120" + ".npy") 
    path_label = "/home/xuemeng/IMU/IMU_Hete/Datasets/Merge/UniHarData" + path_label
    path_data = "/home/xuemeng/IMU/IMU_Hete/Datasets/Merge/UniHarData" + path_data
    print(path_label)
    label = np.load(path_label).astype(np.float32)
    data = np.load(path_data).astype(np.float32)
    user_label_index = 1
    # ["Hhar", "Uci", "Motion", "Shoaib"] 
    # domain 1_9: 0-8; 2_30: 9-38; 3_24: 39-62; 4_10: 63-72 ; 0_73: 0-72
    UserN= check_domain_user_num(label, 2, label_user_index=1, label_domain_index=2, domain_num=4) #domian=0, 表示读取全部的数据；
    print(UserN) 
    
    if ("uci" in args.dataset): # Inner single dataset.
        User_list= np.array([i for i in range(9, 39)]) # User list. [i for i in range(9, 39)] [64,66,67,70,72]
        data_set, label_set = load_Uci_users(User_list, data, label)
        result_tvt= separate_user_tvt(data, data_set, label_set)
        splitTrain_again = 1 
        data_train, label_train, data_vali, label_vali, data_test, label_test \
            = further_split_train(data, splitTrain_again, result_tvt)
        
    elif("shoaib" in args.dataset): # Inter multiple dataset. 
        # Read user_list for test: 
        User_list_1= np.array([i for i in range(0, 8)]) # hhar 2.  single dataset. 
        data_set_1, label_set_1 = load_Uci_users(User_list_1, data, label)
        result_tvt_1= separate_user_tvt(data, data_set_1, label_set_1)
        splitTrain_again = 1 
        # shape is: data_train, label_train, data_vali, label_vali, data_test, label_test
        data_test, label_test, data_vali, label_vali, _, _ \
            = further_split_train(data, splitTrain_again, result_tvt_1)
        
        # Read user_list for train:
        User_list_2= np.array([i for i in range(9, 72)]) # Other three datasets. 
        data_set_2, label_set_2 = load_Across_users(User_list_2, data, label)
        result_tvt_2= separate_user_tvt(data, data_set_2, label_set_2)
        # shape is: data_train, label_train, data_vali, label_vali, data_test, label_test
        data_train, label_train, _,_, _, _ \
            = further_split_train(data, splitTrain_again, result_tvt_2)
        # 注意不同数据集的 搭配，特别是 训练模型的指标
    
    
    # data_train, label_train, data_vali, label_vali, data_test, label_test = partition_and_reshape(data, label)
    
    #if you need to train with part of labeled data, you can use the following code:
    # data_train, label_train, _, _  = prepare_simple_dataset(data_train, label_train,training_rate=1)
    # pipeline = [Preprocess4Normalization(args.input)]
    pipeline=[]
    pipeline_tta = [Preprocess4Normalization(args.input),Preprocess4Sample(args.seq_len, temporal=0.4)
            , Preprocess4Rotation(), Preprocess4Noise(), Preprocess4Permute()]
    if args.model != 'deepsense':
        data_set_train = IMUDataset(data_train, label_train, pipeline=pipeline)
        data_set_test = IMUDataset(data_test, label_test, pipeline=pipeline)
        data_set_vali = IMUDataset(data_vali, label_vali, pipeline=pipeline)
        data_set_tta = IMUTTADataset(data_test, label_test, pipeline=pipeline_tta) # data aug is wrong. 
    else:
        data_set_train = FFTDataset(data_train, label_train, pipeline=pipeline)
        data_set_test = FFTDataset(data_test, label_test, pipeline=pipeline)
        data_set_vali = FFTDataset(data_vali, label_vali, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=args.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=args.batch_size)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=args.batch_size)
    data_loader_tta = DataLoader(data_set_tta, shuffle=False, batch_size=args.batch_size)
    return data_loader_train, data_loader_vali, data_loader_test, data_loader_tta
