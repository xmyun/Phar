import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import copy
# import train
# from config import load_dataset_label_names
# from models import fetch_classifier
# from plot import plot_matrix
from torchattacks import PGD, FGSM
# from utils import *

from argParse import *
from datasetPre import * 

# from IMU_Hete.Src.model_bak1 import *
from model import *


def eval(model,args, data_loader_test, data_loader_tta,epoch):
    """ Evaluation Loop """
    model.train() # evaluation mode
    model = model.to(args.device)
    if args.data_parallel: # use Data Parallelism with Multi-GPU 
        model = nn.DataParallel(model)
    results = [] # prediction results
    labels = [] 
    time_sum = 0.0 
    for batch, batch_tta in zip(data_loader_test, data_loader_test): # Same dataset data_loader_tta 
        batch = [t.to(args.device) for t in batch]
        # batch_tta = [t.to(args.device) for t in batch_tta]
        with torch.no_grad(): # evaluation without gradient calculation
            start_time = time.time()
            inputs, label = batch
            # inputs_tta, label_tta = batch_tta
            result = model.forward_tta(inputs, inputs, epoch) # model inputs_tta forward_tta  inputs, inputs,epoch
            time_sum += time.time() - start_time
            results.append(result)
            labels.append(label)

    label = torch.cat(labels, 0)
    predict = torch.cat(results, 0)
    return stat_acc_f1(label.cpu().numpy(), predict.cpu().numpy())

def cotta(args):
        data_loader_train ,data_loader_valid, data_loader_test, data_loader_tta = load_dataset(args)
        criterion = nn.CrossEntropyLoss()
        """ Train Loop """
        base_model = fetch_classifier(args)
        base_model_dicts = torch.load(args.save_path + args.dataset + '.pt') # shoaib_20_120
        print("路径", args.save_path + args.dataset + '.pt')
        base_model.load_state_dict(base_model_dicts)
        optimizer = torch.optim.Adam(params=base_model.parameters(), lr=args.lr)  # , weight_decay=0.95
        cotta_model = CoTTA_attack_softmatch(model=base_model,optimizer=optimizer,arg=args) # from CoTTA to CoTTA_attack;  CoTTA_attack_softmatch
        # cotta_model = base_model
        cotta_model = cotta_model.to(args.device)
        if args.data_parallel: # use Data Parallelism with Multi-GPU
            cotta_model = nn.DataParallel(cotta_model)
        global_step = 0 # global iteration steps regardless of epochs

        for e in range(args.epoch):
            # train_acc, train_f1 = eval(cotta_model,args, data_loader_train)
            # valid是同一个数据集，以保证训练; test是跨数据集，来测试跨数据集的效果；
            test_acc, test_f1 = eval(cotta_model,args, data_loader_test, data_loader_tta, e)
            # vali_acc, vali_f1 = eval(cotta_model,args, data_loader_valid)
            print('Epoch %d/%d , Accuracy: %0.3f, F1: %0.3f'
                  % (e+1, args.epoch, test_acc,  test_f1)) 
        print('The Total Epoch have been reached.')
        # print('Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f' % best_stat)

if __name__ == "__main__":
    args = set_arg()
    cotta(args)