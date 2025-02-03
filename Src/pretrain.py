import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import copy

from argParse import *
from datasetPre import * 
from model import *

def load(model, device, model_file, load_self=False):
    """ load saved model or pretrained transformer (a part of model) """
    if model_file:
        if load_self:
            model.load_self(model_file + '.pt', map_location=device)
        else:
            model.load_state_dict(torch.load(model_file + '.pt', map_location=device), strict=False)

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

def pretrain(args): 
        data_loader_train ,data_loader_valid,data_loader_test = load_dataset(args) # ,_
        # print("Exit for code debug.") 
        # exit() 
        criterion = nn.CrossEntropyLoss()
        device = get_device(args.g)
        print(device)
        """ Train Loop """
        # self.load(model_file, load_self)
        # model = fetch_classifier(args)
        
        # classifier_cfg = args.model_cfg
        # Pure fetch. 
        model = fetch_classifier(args)
        # classifier = fetch_classifier(args, input=args.encoder_cfg.hidden, output=args.activity_label_size) # output=label_num
        # model = CompositeClassifier(args.encoder_cfg, classifier=classifier) 
        
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)  # , weight_decay=0.95
        
        # load(model, device, model_file=args.e, load_self=True) # Add feature extractor.  --encoder_file
        
        # # Add code for continual train
        # model_continual_dicts = torch.load(args.save_path+"/"+ args.SDom +args.TDom + args.g +"/" + args.dataset+ str(445) + '.pt') 
        # model.load_state_dict(model_continual_dicts)
        
        model = model.to(device)
        
        # if args.data_parallel: # use Data Parallelism with Multi-GPU
        #     model = nn.DataParallel(model)
        
        global_step = 0 # global iteration steps regardless of epochs
        best_stat = None
        
        model_best = model.state_dict()
        loss_list = []
        acc_list = []
        subexp=args.SDom +args.TDom
        subexpFolder= args.save_path + "/" + subexp + args.g + "/"
        if not os.path.exists(subexpFolder):
            # not exit folder, create it!
            os.makedirs(subexpFolder)
        
        for e in range(0, args.Pre_epoch): # range(args.Pre_epoch) 
            loss_sum = 0.0 # the sum of iteration losses to get average loss in every epoch
            time_sum = 0.0
            
            Val_best=0.0
            model.train()
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(device) for t in batch]
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
            loss_rank = sorted_list.index(loss_sum)+50
            train_acc, train_f1 = eval(model,args, data_loader_train) 
            test_acc, test_f1 = eval(model,args, data_loader_test) 
            vali_acc, vali_f1 = eval(model,args, data_loader_valid)
            acc_list.append(vali_acc) 
            sorted_acc = sorted(acc_list, reverse=True)
            acc_rank = sorted_acc.index(vali_acc)+1 
            print('Epoch %d/%d : Average Loss %5.4f, Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f'
                  % (e+1, args.Pre_epoch, loss_sum / len(data_loader_train), train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1))
            best_stat = (train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
            if acc_rank <= 50 :
                model_best = copy.deepcopy(model.state_dict()) 
                torch.save(model.state_dict(),  subexpFolder + args.dataset + str(acc_rank) + '.pt') #  model.state_dict(),  subexpFolder + args.dataset + str(rank) + '.pt'
            if loss_rank <= 100:
                model_best = copy.deepcopy(model.state_dict()) 
                torch.save(model.state_dict(),  subexpFolder + args.dataset + str(loss_rank) + '.pt')
            # model_best = copy.deepcopy(model.state_dict())
            # if vali_acc > Val_best: 
            #     Val_best = vali_acc
            #     torch.save(model.state_dict(),  subexpFolder + args.dataset + "bestval" + '.pt') 
            # # if e>450: # after 450;  
            # #     torch.save(model.state_dict(),  subexpFolder + args.dataset + str(e) + '.pt') 
            # torch.save(model.state_dict(), subexpFolder + args.dataset  + str(rank) + '.pt')
        model.load_state_dict(model_best) 
        print('The Total Epoch have been reached.')
        print('Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f' % best_stat)

if __name__ == "__main__":
    args = set_arg()
    print(args)
    set_seeds(args.seed)
    print("Seed number in Train:", args.seed)
    pretrain(args)