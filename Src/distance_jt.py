import torch
import numpy as np
import torch.nn.functional as F
import os
import time
from torch import nn
# from datasetPre import * 
# from argParse import *
# from utils import Preprocess4Sample_t,Preprocess4Rotation_t,Preprocess4Normalization_t, Preprocess4Noise_t, Preprocess4Permute_t
# from torchattacks import PGD

# from model import *

def kmmd_dist(x1, x2):
    X_total = torch.cat([x1,x2],0)
    Gram_matrix = gaussian_kernel(X_total,X_total,kernel_mul=2.0, kernel_num=2, fix_sigma=0, mean_sigma=0)
    n = int(x1.shape[0])
    m = int(x2.shape[0])
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
    if fix_sigma:
        bandwidth = fix_sigma
    elif mean_sigma:
        bandwidth = torch.mean(L2_distance)
    else:  ## median distance
        bandwidth = torch.mean(L2_distance)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  #

def J_t(model, source, target):

    original_feature = []
    augmented_feature = []
    original_feature = model(source)
    augmented_feature = model(target)
    feature_test = torch.cat([original_feature, augmented_feature], 0)
    original_label_test = torch.zeros((original_feature.shape[0],), device=source.device)
    augmented_label_test = torch.ones((augmented_feature.shape[0],), device=source.device)
    label_test = torch.cat([original_label_test, augmented_label_test], 0)
    ood_detection = Feature_Correlations(POWER_list=torch.arange(1, 9), mode='mad')
    ood_detection.train(in_data=[original_feature])
    threshold_95, threshold_99 = threshold_determine(original_feature, ood_detection)
    test_deviations = ood_detection.get_deviations_([feature_test])
    ood_label_95 = torch.where(test_deviations > threshold_95, 1, 0).squeeze()
    ood_label_99 = torch.where(test_deviations > threshold_99, 1, 0).squeeze()
    clean_feature_group = feature_test[ood_label_95 == 0]
    bd_feature_group = feature_test[ood_label_95 == 1]
    clean_feature_flat = torch.mean(clean_feature_group,dim=1)
    bd_feature_flat = torch.mean(bd_feature_group,dim=1)
    if bd_feature_flat.shape[0] < 1:
        kmmd = torch.Tensor([0])
        kmmd.requires_grad = True
    else:
        kmmd = kmmd_dist(clean_feature_flat, bd_feature_flat)
    
    return kmmd.to(source.device)

def threshold_determine(clean_feature_target, ood_detection):
    test_deviations_list = []
    step = 5
    # print(len(clean_feature_target))
    for i in range(step):
        index_mask = np.ones((len(clean_feature_target),))
        index_mask[i*int(len(clean_feature_target)//step):(i+1)*int(len(clean_feature_target)//step)] = 0
        clean_feature_target_train= clean_feature_target[np.where(index_mask == 1)]
        clean_feature_target_test = clean_feature_target[np.where(index_mask == 0)]
        if len(clean_feature_target_test) < 1:
            continue
        ood_detection.train(in_data=[clean_feature_target_train],)
        test_deviations = ood_detection.get_deviations_([clean_feature_target_test])
        test_deviations_list.append(test_deviations)
    test_deviations = torch.cat(test_deviations_list,0)
    test_deviations_sort = torch.sort(test_deviations,0)

    percentile_95 = test_deviations_sort[int(len(test_deviations_sort)*0.95)][0]
    percentile_99 = test_deviations_sort[int(len(test_deviations_sort)*0.99)][0]

    return percentile_95, percentile_99

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
        # print(temp.shape)
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
            batch_deviations.append(dev)
        batch_deviations = torch.cat(batch_deviations, dim=1)
        deviations.append(batch_deviations)
        deviations = torch.cat(deviations, dim=0) / (self.num_feature * len(self.power))
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
