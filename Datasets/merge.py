# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np


def transform_label(label_act, act_label_indexes, label_user, domain_num, dataset_index):
    label_act_new = np.zeros(label_act.shape)
    for i in range(len(act_label_indexes)):
        if isinstance(act_label_indexes[i], list):
            for l in act_label_indexes[i]:
                index = label_act == l
                label_act_new[index] = i
        else:
            index = label_act == act_label_indexes[i]
            label_act_new[index] = i
    if len(label_user.shape) > 2:
        label_user_unique = np.unique(label_user[:, 0, :], axis=0)
        label_user = np.concatenate([np.where(np.all(label_user[i, 0, :] == label_user_unique, axis=1))
                                     for i in range(label_user.shape[0])])
        label_user = np.repeat(label_user, label_act.shape[1], axis=1)
    #print(np.unique(label_user[:,0]).size)
    label_user = label_user + domain_num
    label_domain = np.ones(label_act.shape) * dataset_index
    label_new = np.array([label_act_new, label_user, label_domain]).transpose([1, 2, 0])
    return label_new


def merge(datasets, versions, activity_label_indexes, activity_indexes, user_indexes, feature_num=6):
    data_merge = []
    label_merge = []
    domain_num = 0
    for d in range(len(datasets)):
        path_data = os.path.join(datasets[d], "data_" + versions[d] + ".npy")
        path_label = os.path.join(datasets[d], "label_" + versions[d] + ".npy")
        data = np.load(path_data).astype(np.float32)
        label = np.load(path_label).astype(np.float32)
        # print(path_data)
        print(data.shape)
        print(label.shape)
        # print(np.unique(np.array(label[:,:,1])))
        # print(np.max(data[:,:,:]))
        label_new = transform_label(label[:, :, activity_indexes[d]], activity_label_indexes[d]
                                    , label[:, :, user_indexes[d]], domain_num, d)
        domain_num = np.max(label_new[:, :, 1]) + 1
        data_merge.append(data[:, :, :feature_num])
        label_merge.append(label_new)
        #print(np.unique(label_new[:,:,1]))
    data_merge = np.concatenate(data_merge, 0)
    label_merge = np.concatenate(label_merge, 0)
    return data_merge, label_merge


if __name__ == "__main__":
    datasets = ["hhar", "uci", "motion", "shoaib",'usc-had','ku-har'] #
    versions = ["20_120"] * 6
    domain_indexes = [0, 1, 1, 2, 1, 1] # user dimension
    activity_indexes = [2, 0, 0, 0, 0, 0]
    # still, walking, walking upstairs, walking downstairs, jogging, bike
    activity_label_indexes = [[[1, 4], 5, 3, 2, -1, 0]
        , [[3, 4, 5], 0, 1, 2, -1]
        , [[2, 3], 4, 1, 0, 5, -1]
        , [[1, 2], 0, 5, 6, 3, 4]
        , [[7, 8, 9], [0, 1, 2], 3, 4,  5, [6, 11, 12]]
        , [[0, 1, 5], [11, 13], 15, 16]
        ]
    data, label = merge(datasets, versions, activity_label_indexes, activity_indexes, domain_indexes)

    # np.save(os.path.join("new", "data_" + versions[0] + ".npy"), data)
    # np.save(os.path.join("new", "label_" + versions[0] + ".npy"), label)
    print("After merged:")
    print(data.shape)
    print(label.shape)
    # num1, num2, num3, num4, num5, num6= 9166, 2088, 4534, 10500, 4350, 3892
