import torch
import numpy as np
import torch.nn.functional as F
import os
import time
from torch import nn
from datasetPre import * 
from argParse import * 
from torch.serialization import MAP_LOCATION 
from utils import Preprocess4Sample_t,Preprocess4Rotation_t,Preprocess4Normalization_t, Preprocess4Noise_t, Preprocess4Permute_t
from torchattacks import PGD 
from itertools import chain 
from model import * 
from geomloss import SamplesLoss
from scipy.stats import wasserstein_distance
from model import PGD1

def kmmd_dist(x1, x2):
    X_total = torch.cat([x1,x2],0)
    Gram_matrix = gaussian_kernel(X_total,X_total,kernel_mul=2.0, kernel_num=2, fix_sigma=0, mean_sigma=0)
    n = int(x1.shape[0])
    m = int(x2.shape[0])
    x1x1 = Gram_matrix[:n, :n]
    x2x2 = Gram_matrix[n:, n:]
    x1x2 = Gram_matrix[:n, n:]
    # MMD calculation
    mmd = torch.mean(x1x1) + torch.mean(x2x2) - 2 * torch.mean(x1x2)
    normalization_factor = (m * n) / (m + n)
    mmd = normalization_factor * mmd
    penalty_P = 1 * torch.mean(x1x1)
    penalty_Q = 1 * torch.mean(x2x2)
    # Adjusted MMD
    diff = mmd - penalty_P - penalty_Q
    
    # # 将 Gram 矩阵展平为一维数组
    # x1x1_flat = x1x1.cpu().numpy().flatten()
    # x2x2_flat = x2x2.cpu().numpy().flatten()
    # x1x2_flat = x1x2.cpu().numpy().flatten()
    # wasserstein_distance_x1 = wasserstein_distance(x1x1_flat, x1x2_flat)
    # wasserstein_distance_x2 = wasserstein_distance(x2x2_flat, x1x2_flat)
    # diff = (wasserstein_distance_x1 + wasserstein_distance_x2) / 2
    # diff = torch.tensor(diff).to(x1.device)# 将融合后的 Wasserstein 距离转换为 PyTorch 张量
    
    print('diff xuemeng', diff)
    return diff


def kmmd_dist_back1(x1, x2):
    
    X_total = torch.cat([x1,x2],0)
    Gram_matrix = gaussian_kernel(X_total,X_total,kernel_mul=2.0, kernel_num=2, fix_sigma=0, mean_sigma=0)
    n = int(x1.shape[0])
    m = int(x2.shape[0])
    diff = (m*n)/(m+n)*Gram_matrix
    print('shape1 is ',x1.shape)
    print('shape2 is ',x2.shape)
    print('gram_diff is :',diff)
    max_len = max(m,n)
    if len(x1) < max_len:
      x1 = torch.nn.functional.pad(x1, (0, max_len - len(x1)))
    if len(x2) < max_len:
      x2 = torch.nn.functional.pad(x2, (0, max_len - len(x2)))
    w1 = torch.fft.fft(x1)
    w2 = torch.fft.fft(x2)
    phase1 = torch.angle(w1)
    phase2 = torch.angle(w2)
    phase_diff = phase1 - phase2
    abs_phase_diff = torch.abs(phase_diff)
    # 2. 求绝对值的均值
    phase_diff = torch.mean(abs_phase_diff)
    print('phase_diff is ', phase_diff)
    return diff + phase_diff

def gaussian_kernel(x1, x2, kernel_mul=2.0, kernel_num=5, fix_sigma=0, mean_sigma=0):
    x1_sample_size = x1.shape[0]
    x2_sample_size = x2.shape[0]
    # print(x1_sample_size,x2_sample_size)
    # print(x1.shape,x2.shape)
    
    #new version for gram matrix calculate.
    x1_tile_shape = [x1_sample_size, 1]
    x2_tile_shape = [1, x2_sample_size] 
    tile_x1 = x1.view(x1_tile_shape).repeat(1, x2_sample_size)
    tile_x2 = x2.view(x2_tile_shape).repeat(x1_sample_size, 1)
    L2_distance = torch.square(tile_x1 - tile_x2) 
    #new version for gram matrix calculate.
    
    # Old version for gram matrix calculate. 
    # x1_tile_shape = []
    # x2_tile_shape = []
    # norm_shape = []
    # for i in range(len(x1.shape) + 1):
    #     if i == 1:
    #         x1_tile_shape.append(x2_sample_size)
    #     else:
    #         x1_tile_shape.append(1)
    #     if i == 0:
    #         x2_tile_shape.append(x1_sample_size)
    #     else:
    #         x2_tile_shape.append(1)
    #     if not (i == 0 or i == 1):
    #         norm_shape.append(i)

    # tile_x1 = torch.unsqueeze(x1, 1).repeat(x1_tile_shape)
    # tile_x2 = torch.unsqueeze(x2, 0).repeat(x2_tile_shape)
    # L2_distance = torch.square(tile_x1 - tile_x2).sum(dim=norm_shape)
    # Old version for gram matrix calculate. 
    if fix_sigma:
        bandwidth = fix_sigma
    elif mean_sigma:
        bandwidth = torch.mean(L2_distance)
    else:  ## median distance
        # bandwidth = torch.mean(L2_distance) # Old version for gram matrix calculate. 
         bandwidth = torch.median(L2_distance) # new version for gram matrix calculate.
         
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  #L2_distance #

def tta_transform(arg):
    return [Preprocess4Normalization_t(arg.model_cfg.feature_num),Preprocess4Sample_t(arg.seq_len, temporal=0.4) 
                          ,Preprocess4Rotation_t(), Preprocess4Noise_t(), Preprocess4Permute_t()]# arg.input
    
def J_t(model, source, target):
    # J_t = []
    
    # def hook_fn(module, input, output):
    #     # print("Hook called!")
    #     # print("Module:", module)
    #     # print("Input:", input)
    #     # print("Output:", output)
    #     return
    # hook = model.lin1.register_forward_hook(hook_fn)
    
    original_feature = []
    augmented_feature = []
    original_feature = model(source)
    augmented_feature = model(target)
    # print(original_feature[0].shape)
    feature_test = torch.cat([original_feature, augmented_feature], 0)
    original_label_test = torch.zeros((original_feature.shape[0],), device=source.device)
    augmented_label_test = torch.ones((augmented_feature.shape[0],), device=source.device)
    label_test = torch.cat([original_label_test, augmented_label_test], 0)
    ood_detection = Feature_Correlations(POWER_list=torch.arange(1, 9), mode='mad')
    ood_detection.train(in_data=[original_feature])
    original_deviations_sort = torch.sort(ood_detection.get_deviations_([original_feature]), 0)[0]
    augmented_deviations_sort = torch.sort(ood_detection.get_deviations_([augmented_feature]), 0)[0]
    percentile_95 = torch.where(augmented_deviations_sort > original_deviations_sort[int(len(original_deviations_sort) * 0.95)], 1, 0)
    # print(f'percentile_95:{clean_deviations_sort[int(len(clean_deviations_sort)*0.95)]},TP95:{percentile_95.sum()/len(bd_deviations_sort)}')
    percentile_99 = torch.where(augmented_deviations_sort > original_deviations_sort[int(len(original_deviations_sort) * 0.99)], 1, 0)
    # print(f'percentile_99:{clean_deviations_sort[int(len(clean_deviations_sort)*0.99)]},TP99:{percentile_99.sum()/len(bd_deviations_sort)}')
    # print('original_feature:',len(original_feature))
    threshold_95, threshold_99 = threshold_determine(original_feature, ood_detection)
    test_deviations = ood_detection.get_deviations_([feature_test])
    ood_label_95 = torch.where(test_deviations > threshold_95, 1, 0).squeeze()
    ood_label_99 = torch.where(test_deviations > threshold_99, 1, 0).squeeze()
    false_negetive_95 = torch.where(label_test - ood_label_95 > 0, 1, 0).squeeze()
    false_negetive_99 = torch.where(label_test - ood_label_99 > 0, 1, 0).squeeze()
    false_positive_95 = torch.where(label_test - ood_label_95 < 0, 1, 0).squeeze()
    false_positive_99 = torch.where(label_test - ood_label_99 < 0, 1, 0).squeeze()
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
        # print(clean_feature_target_test)
        if len(clean_feature_target_test) < 1:
            continue
        ood_detection.train(in_data=[clean_feature_target_train],)
        test_deviations = ood_detection.get_deviations_([clean_feature_target_test])
        test_deviations_list.append(test_deviations)
    # test_deviations = np.concatenate(test_deviations_list,0)
    test_deviations = torch.cat(test_deviations_list,0)
    test_deviations_sort = torch.sort(test_deviations,0)

    percentile_95 = test_deviations_sort[int(len(test_deviations_sort)*0.95)][0]
    percentile_99 = test_deviations_sort[int(len(test_deviations_sort)*0.99)][0]
    # print(f'percentile_95:{percentile_95}')
    # print(f'percentile_99:{percentile_99}')
    # plt.hist(test_deviations)
    # plt.show()
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
        # batch_deviations = np.concatenate(batch_deviations,axis=1)
        batch_deviations = torch.cat(batch_deviations, dim=1)
        deviations.append(batch_deviations)
        # deviations = np.concatenate(deviations,axis=0) /self.num_feature /len(self.power)
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

def eval(model,args, data_loader_test):
    """ Evaluation Loop """
    model.eval() # evaluation mode 
    device = get_device(args.g)
    model = model.to(device) 
    if args.data_parallel: # use Data Parallelism with Multi-GPU
        model = nn.DataParallel(model) 
    results = [] # prediction results
    labels = []
    time_sum = 0.0
    for batch in data_loader_test:
        batch = [t.to(device) for t in batch]
        with torch.no_grad(): # evaluation without gradient calculation
            start_time = time.time()
            inputs, label = batch
            result = model(inputs, False) 
            time_sum += time.time() - start_time
            results.append(result)
            labels.append(label)
    label = torch.cat(labels, 0)
    predict = torch.cat(results, 0)
    return stat_acc_f1(label.cpu().numpy(), predict.cpu().numpy())

def eval_gram(model, args, data_loader_validate):
    """ Evaluation Loop """
    transforms = tta_transform(args)
    model.eval()  # evaluation mode   eval
    device = get_device(args.g)
    model = model.to(device)
    if args.data_parallel:  # use Data Parallelism with Multi-GPU
        model = nn.DataParallel(model)
    results = []  # prediction results 
    time_sum = 0.0
    model.train() # aes
    attack = PGD1(model, eps=0.01, alpha=1 / 255, steps=1) # aes Step 10. 8/255 
    for batch_valid in data_loader_validate: 
        all_Jt = [] 
        batch_valid = [t.to(device) for t in batch_valid]
        inputs_t, label_t = batch_valid 
        outputs_aes = model(inputs_t) 
        input = attack.attack(inputs_t, outputs_aes)  #  aes 
        J_t_median = J_t(model, inputs_t, input) #  aes 
        all_Jt.append(J_t_median.to(device)) #  aes
        with torch.no_grad():  # evaluation without gradient calculation
            start_time = time.time()
            for transform in transforms:
                input = transform(inputs_t.clone())
                J_t_median = J_t(model, inputs_t, input)
                all_Jt.append(J_t_median.to(device))
            all_Jt = torch.tensor(all_Jt)
            J_t_median = torch.mean(all_Jt)
            time_sum += time.time() - start_time
            results.append(J_t_median.detach())
    results_tensor = torch.tensor(results)
    median_result = torch.sum(results_tensor)
    print('results:', median_result.item())
    return median_result.item()

# from fetch_model import fetch_classifier
def select_model(args,source,target):
    J_t = []
    acc = []
    subexp=args.SDom +args.TDom
    # subexpFloder= 'checkpoint_singletosingle'+"/"+subexp+ '2' +"/"
    subexpFloder= 'checkpoint'+"/"+subexp+ '2' + "/"  # motion2hhar: 2
    # 检查文件夹是否存在
    if not os.path.exists(subexpFloder):
        # 文件夹不存在，创建文件夹
        # os.makedirs(subexpFloder)
        print("There is no pretrained model!")
        
    for i in range(11, 21): # 699 11-21? or 1-11? 1-21?
    # for i in chain(range(1,11),range(51,62)): # last hun
        # model = fetch_classifier(args)
        classifier = fetch_classifier(args, input=args.encoder_cfg.hidden, output=args.activity_label_size) # output=label_num
        model = CompositeClassifier(args.encoder_cfg, classifier=classifier) 
        
        device = get_device(args.g)
        model_path = subexpFloder + args.dataset+ str(i) + '.pt' 
        print(model_path)
        if os.path.exists(model_path):
            print(model_path)
            model_dicts = torch.load(model_path,map_location=device)
            model.load_state_dict(model_dicts)
            model = model.to(device)
            model.train() # eval
            # test_acc,test_F1 = eval(model,args, target)
            # acc.append([test_acc,model_path])
            J_t_median = eval_gram(model,args, target)
            J_t.append([J_t_median,model_path])
            # print(f'J_t_median:{J_t_median}')
    # acc_list = sorted(acc,key=lambda x:x[0])
    classifier = fetch_classifier(args, input=args.encoder_cfg.hidden, output=args.activity_label_size) # output=label_num
    model = CompositeClassifier(args.encoder_cfg, classifier=classifier) 
    device = get_device(args.g)
    model_path = subexpFloder + args.dataset+ 'bestval' + '.pt' 
    print(model_path)
    if os.path.exists(model_path):
        print(model_path)
        model_dicts = torch.load(model_path,map_location=device)
        model.load_state_dict(model_dicts)
        model = model.to(device)
        model.train() # eval
        J_t_median = eval_gram(model,args, target)
        J_t.append([J_t_median,model_path])
    
    J_list = sorted(J_t,key=lambda x:x[0])
    print(J_list)
    print(J_list[0][1])
    # return acc_list[699][1]
    for i in J_list:
        if i[0] != 0:
            return i[1]
    return J_list[0][1]
