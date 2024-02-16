import torch
import numpy as np
import torch.nn.functional as F
def kmmd_dist(x1, x2):
    X_total = torch.cat([x1,x2],0)
    Gram_matrix = gaussian_kernel(X_total,X_total,kernel_mul=2.0, kernel_num=2, fix_sigma=0, mean_sigma=0)
    n = int(x1.shape[0])
    m = int(x2.shape[0])
    # print(n,m)
    # print("gram",Gram_matrix)
    # x1x1 = Gram_matrix[:n, :n]
    # x2x2 = Gram_matrix[n:, n:]
    # x1x2 = Gram_matrix[:n, n:]
    # x2x1 = Gram_matrix[n:, :n]  # Gram_matrix is symmetric
    # diff = torch.mean(x1x1) + torch.mean(x2x2) - 2 * torch.mean(x1x2)
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
    # print("l2",L2_distance)
    # bandwidth inference
    # print(L2_distance.shape)
    if fix_sigma:
        bandwidth = fix_sigma
    elif mean_sigma:
        bandwidth = torch.mean(L2_distance)
    else:  ## median distance
        # bandwidth = torch.median(L2_distance.reshape(L2_distance.shape[0],-1))
        # print("L2",L2_distance.shape)
        # bandwidth = torch.median(L2_distance.reshape(L2_distance.shape[0],-1))
        bandwidth = torch.mean(L2_distance)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # print('list',bandwidth_list)
    #print(torch.cat(bandwidth_list,0).to(torch.device('cpu')).numpy())
    ## gaussian_RBF = exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  #L2_distance #


def J_t(model, source, target):
    J_t = []
    def hook_fn(module, input, output):
        # print("Hook called!")
        # print("Module:", module)
        # print("Input:", input)
        # print("Output:", output)
        return
    hook = model.lin1.register_forward_hook(hook_fn)
    original_feature = []
    augmented_feature = []
    original_feature=model(source).to(torch.device('cpu'))
    augmented_feature=model(target).to(torch.device('cpu'))
    # print(original_feature[0].shape)
    feature_test = torch.cat([original_feature,augmented_feature],0)
    original_label_test = np.zeros((original_feature.shape[0],))
    augmented_label_test = np.ones((augmented_feature.shape[0],))
    label_test = np.concatenate([original_label_test,augmented_label_test],0)
    ood_detection = Feature_Correlations(POWER_list=np.arange(1,9),mode='mad')
    ood_detection.train(in_data=[original_feature])
    original_deviations_sort = np.sort(ood_detection.get_deviations_([original_feature]),0)
    augmented_deviations_sort = np.sort(ood_detection.get_deviations_([augmented_feature]),0)
    percentile_95 = np.where(augmented_deviations_sort > original_deviations_sort[int(len(original_deviations_sort)*0.95)], 1, 0)
    # print(f'percentile_95:{clean_deviations_sort[int(len(clean_deviations_sort)*0.95)]},TP95:{percentile_95.sum()/len(bd_deviations_sort)}')
    percentile_99 = np.where(augmented_deviations_sort > original_deviations_sort[int(len(original_deviations_sort)*0.99)], 1, 0)
    # print(f'percentile_99:{clean_deviations_sort[int(len(clean_deviations_sort)*0.99)]},TP99:{percentile_99.sum()/len(bd_deviations_sort)}')
    threshold_95, threshold_99 = threshold_determine(original_feature, ood_detection)
    test_deviations = ood_detection.get_deviations_([feature_test])
    ood_label_95 = np.where(test_deviations > threshold_95, 1, 0).squeeze()
    ood_label_99 = np.where(test_deviations > threshold_99, 1, 0).squeeze()
    false_negetive_95 = np.where(label_test - ood_label_95 > 0, 1, 0).squeeze()
    false_negetive_99 = np.where(label_test - ood_label_99 > 0, 1, 0).squeeze()
    false_positive_95 = np.where(label_test - ood_label_95 < 0, 1, 0).squeeze()
    false_positive_99 = np.where(label_test - ood_label_99 < 0, 1, 0).squeeze()
    # print(f'false_negetive_95:{false_negetive_95.sum()},false_negetive_99:{false_negetive_99.sum()}')
    # print(f'false_positive_95:{false_positive_95.sum()},false_positive_99：{false_positive_99.sum()}')
    clean_feature_group = feature_test[np.where(ood_label_95==0)]
    bd_feature_group = feature_test[np.where(ood_label_95==1)]
    # clean_feature_flat = torch.mean(clean_feature_group,dim=(2,3))
    # bd_feature_flat = torch.mean(bd_feature_group,dim=(2,3))
    # print(clean_feature_group.shape)
    clean_feature_flat = torch.mean(clean_feature_group,dim=1)
    bd_feature_flat = torch.mean(bd_feature_group,dim=1)
    if bd_feature_flat.shape[0] < 1:
        kmmd = torch.Tensor([0])
        kmmd.requires_grad = True
        # kmmd.grad_fn = torch.mul(kmmd, 1)
    else:
        kmmd = kmmd_dist(clean_feature_flat, bd_feature_flat)
    # J_t.append(kmmd.item())
    # J_t = np.asarray(J_t)
    # J_t_median = np.median(J_t)
    # J_MAD = np.median(np.abs(J_t - J_t_median))
    # J_star = np.abs(J_t - J_t_median)/1.4826/(J_MAD+1e-6)
    # print('J_t_median:',J_t_median)
    return kmmd
def threshold_determine(clean_feature_target, ood_detection):
    test_deviations_list = []
    step = 5
    for i in range(step):
        index_mask = np.ones((len(clean_feature_target),))
        index_mask[i*int(len(clean_feature_target)//step):(i+1)*int(len(clean_feature_target)//step)] = 0
        clean_feature_target_train= clean_feature_target[np.where(index_mask == 1)]
        clean_feature_target_test = clean_feature_target[np.where(index_mask == 0)]
        ood_detection.train(in_data=[clean_feature_target_train],)
        test_deviations = ood_detection.get_deviations_([clean_feature_target_test])
        test_deviations_list.append(test_deviations)
    test_deviations = np.concatenate(test_deviations_list,0)
    test_deviations_sort = np.sort(test_deviations,0)
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

        temp = temp.reshape(temp.shape[0], temp.shape[1], -1) #temp.shape[0], temp.shape[1], -1
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
            if (feat_L.detach()).shape[0] == 0: # zero will present bugs. 
                    continue
            if L==len(medians):
                medians.append([None]*len(self.power))
                mads.append([None]*len(self.power))
            for p,P in enumerate(self.power):
                # print((feat_L.detach()**P).shape[0])
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
            batch_deviations.append(dev.cpu().detach().numpy())
        batch_deviations = np.concatenate(batch_deviations,axis=1)
        deviations.append(batch_deviations)
        deviations = np.concatenate(deviations,axis=0) /self.num_feature /len(self.power)
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
