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
from gram import select_model
from model import *
from fetch_model import fetch_classifier
# from modelbak import *

def eval(model,args, data_loader_test,data_loader_tta,epoch):
    """ Evaluation Loop """
    model.train() # evaluation mode
    # self.load(model_file, load_self=load_self)
    # print(count_model_parameters(self.model))
    model = model.to(args.device)
    # intermedia_feature = LayerActivations(model)
    if args.data_parallel: # use Data Parallelism with Multi-GPU
        model = nn.DataParallel(model)
    results = [] # prediction results
    labels = []
    original_feature = []
    augmented_feature = []
    time_sum = 0.0
    for batch,batch_tta in zip(data_loader_test,data_loader_test):
        batch = [t.to(args.device) for t in batch]
        batch_tta = [t.to(args.device) for t in batch_tta]
        with torch.no_grad(): # evaluation without gradient calculation
            start_time = time.time()
            inputs, label = batch
            inputs_tta, label_tta = batch_tta
            # inputs_source, mask_inputs_source, masked_pos_source, inputs_selection_source, label = batch_tta
            # print(len(batch_tta))
            # print(inputs_source.shape)
            # print(mask_inputs_source.shape)
            # print(masked_pos_source.shape)
            # print(inputs_selection_source.shape)
            # torch.Size([128, 20, 6])
            # torch.Size([128, 20, 6])
            # torch.Size([128, 3])
            # torch.Size([128, 3, 6])
            # torch.Size([128, 20, 6])
            # torch.Size([20, 6])
            result = model.forward_tta(inputs,inputs,epoch) #  ,inputs_tta 
            # result, label = func_forward(model, batch)
            time_sum += time.time() - start_time
            results.append(result)
            labels.append(label)
    # print("Eval execution time: %.5f seconds" % (time_sum / len(dt)))
    # if func_evaluate:
    #     return func_evaluate(torch.cat(labels, 0), torch.cat(results, 0))
    # else:
    #     return torch.cat(results, 0).cpu().numpy()
    label = torch.cat(labels, 0)
    predict = torch.cat(results, 0)
    return stat_acc_f1(label.cpu().numpy(), predict.cpu().numpy())

def cotta(args):
        data_loader_train ,data_loader_valid,data_loader_test,data_loader_tta = load_dataset(args)
        criterion = nn.CrossEntropyLoss()
        """ Train Loop """
        # self.load(model_file, load_self)
        base_model = fetch_classifier(args)
        # base_model.lin2 = nn.Linear(400, 6)
        best_model_path = select_model(args,data_loader_train,data_loader_test)
        base_model_dicts = torch.load(best_model_path)
        # base_model_dicts = torch.load(args.save_path + "shoaib_20_120best" + '.pt') # args.source_dataset # shoaib
        base_model.load_state_dict(base_model_dicts)
        # base_model.lin2 = nn.Linear(400, 7)
        optimizer = torch.optim.Adam(params=base_model.parameters(), lr=args.lr)  # , weight_decay=0.95
        cotta_model = CoTTA_attack_softmatch(model=base_model,optimizer=optimizer,arg=args)
        # cotta_model = base_model
        cotta_model = cotta_model.to(args.device)
        if args.data_parallel: # use Data Parallelism with Multi-GPU
            cotta_model = nn.DataParallel(cotta_model)
        global_step = 0 # global iteration steps regardless of epochs
        # vali_acc_best = 0.0
        # best_stat = None
        # model_best = model.state_dict()
        for e in range(args.epoch):
            # loss_sum = 0.0 # the sum of iteration losses to get average loss in every epoch
            # time_sum = 0.0
            # cotta_model.train()
            # for i, batch in enumerate(data_loader_train):
            #     batch = [t.to(args.device) for t in batch]
            #     start_time = time.time()
            #     optimizer.zero_grad()
            #     inputs, label = batch
            #     logits = model(inputs, True)
            #     loss = criterion(logits, label)
            #     loss = loss.mean()# mean() for Data Parallelism
            #     loss.backward()
            #     optimizer.step()
            #     global_step += 1
            #     loss_sum += loss.item()
            #     time_sum += time.time() - start_time
            #     # if self.cfg.total_steps and self.cfg.total_steps < global_step:
            #     #     print('The Total Steps have been reached.')
            #         # return
            # train_acc, train_f1 = eval(cotta_model,args, data_loader_train)
            # valid是同一个数据集，以保证训练; test是跨数据集，来测试跨数据集的效果；
            test_acc, test_f1 = eval(cotta_model,args, data_loader_test, data_loader_tta,e)
            # vali_acc, vali_f1 = eval(cotta_model,args, data_loader_valid)
            print('Epoch %d/%d , Accuracy: %0.3f, F1: %0.3f'
                  % (e+1, args.epoch, test_acc,  test_f1))
            # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))
        #     if vali_acc > vali_acc_best:
        #         vali_acc_best = vali_acc
        #         best_stat = (train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
        #         model_best = copy.deepcopy(model.state_dict())
        #         torch.save(model.state_dict(),  args.save_path + '.pt')
        # model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')
        # print('Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f' % best_stat)

if __name__ == "__main__":
    args = set_arg()
    cotta(args)