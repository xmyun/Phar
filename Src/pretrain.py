import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import copy

from argParse import *
from datasetPre import * 
from model import *

def eval(model,args, data_loader_test):
    """ Evaluation Loop """
    model.eval() # evaluation mode
    model = model.to(args.device)
    if args.data_parallel: # use Data Parallelism with Multi-GPU
        model = nn.DataParallel(model)
    results = [] # prediction results
    labels = []
    time_sum = 0.0
    for batch in data_loader_test:
        batch = [t.to(args.device) for t in batch]
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

def pretrain(args):
        data_loader_train ,data_loader_valid,data_loader_test,_ = load_dataset(args)
        criterion = nn.CrossEntropyLoss()
        """ Train Loop """
        # self.load(model_file, load_self)
        model = fetch_classifier(args)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)  # , weight_decay=0.95
        model = model.to(args.device)
        if args.data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)
        global_step = 0 # global iteration steps regardless of epochs
        best_stat = None
        model_best = model.state_dict()
        loss_list = []
        for e in range(args.epoch):
            loss_sum = 0.0 # the sum of iteration losses to get average loss in every epoch
            time_sum = 0.0
            model.train()
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(args.device) for t in batch]
                start_time = time.time()
                optimizer.zero_grad()
                inputs, label = batch
                logits = model(inputs, True)
                loss = criterion(logits, label)
                loss = loss.mean()# mean() for Data Parallelism
                loss.backward()
                optimizer.step()
                global_step += 1
                loss_sum += loss.item()
                time_sum += time.time() - start_time
            loss_list.append(loss_sum)
            sorted_list = sorted(loss_list)
            rank = sorted_list.index(loss_sum)+1
            train_acc, train_f1 = eval(model,args, data_loader_train)
            test_acc, test_f1 = eval(model,args, data_loader_test)
            vali_acc, vali_f1 = eval(model,args, data_loader_valid)
            print('Epoch %d/%d : Average Loss %5.4f, Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f'
                  % (e+1, args.epoch, loss_sum / len(data_loader_train), train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1))
            best_stat = (train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
            model_best = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),  args.save_path + args.dataset+ str(rank) + '.pt')
        model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')
        print('Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f' % best_stat)

if __name__ == "__main__":
    args = set_arg()
    pretrain(args)