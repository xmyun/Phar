import numpy as np
import torch 
from torch import init_num_threads, nn 
from torch.utils.data import DataLoader 
import time 
import copy 
from torchattacks import PGD, FGSM
from argParse import * 
from datasetPre import * 
import pickle
from gram import select_model
from model import * 
from scipy.interpolate import CubicSpline
# from fetch_model import fetch_classifier

def eval(model, args, data_loader_test, epoch): # eval(model, args, data_loader_test, data_loader_tta, epoch)
    """ Evaluation Loop """
    model.train() # evaluation mode //////train
    device = get_device(args.g)
    model = model.to(device)
    if args.data_parallel: # use Data Parallelism with Multi-GPU 
        model = nn.DataParallel(model)
    results = [] # prediction results
    labels = [] 
    time_sum = 0.0 
    J_t = []
    for batch in data_loader_test: # Same dataset data_loader_tta 
    # for batch, batch_tta in zip(data_loader_test, data_loader_test): # Same dataset data_loader_tta 
        batch = [t.to(device) for t in batch]
        # batch_tta = [t.to(device) for t in batch_tta]
        with torch.no_grad(): # evaluation without gradient calculation 
            start_time = time.time()
            inputs, label = batch
            # inputs_tta, label_tta = batch_tta
            result,loss,gram_dist = model.forward_tta(inputs, inputs, epoch) # model inputs_tta forward_tta  inputs, inputs,epoch
            J_t.append(gram_dist)
            time_sum += time.time() - start_time
            results.append(result)
            labels.append(label)
    distance = torch.tensor(J_t)
    print('distance_sum:',torch.sum(distance))
    # print('loss:',loss)

    label = torch.cat(labels, 0)
    predict = torch.cat(results, 0)
    return stat_acc_f1(label.cpu().numpy(), predict.cpu().numpy()), torch.sum(distance)

def cotta(args):
        data_loader_train , data_loader_valid, data_loader_test = load_dataset(args) # , data_loader_tta
        # print("Exit for code debug.")
        # exit()
        criterion = nn.CrossEntropyLoss()
        """ Train Loop """
        distance = []
        # base_model = fetch_classifier(args) 
        classifier = fetch_classifier(args, input=args.encoder_cfg.hidden, output=args.activity_label_size) # output=label_num 
        base_model = CompositeClassifier(args.encoder_cfg, classifier=classifier) 
        
        # Model select. 
        best_model_path = select_model(args,data_loader_train,data_loader_test) 
        
        # best_model_path = 'checkpoint/motion3hhar12/new_20_120bestval.pt' # 测试指定模型
        # best_model_path = args.save_path+"/"+ args.SDom +args.TDom + args.g +"/" + args.dataset+ str(443) + '.pt' # model number. 7:no extract; 
        # best_model_path = args.save_path+"/"+ args.SDom +args.TDom + '0' +"/" + args.dataset + str(3) + '.pt' # model number. 7:no extract; 
        print("The selected model is", best_model_path) 
        
        # Using the selected model. 
        device = get_device(args.g)
        base_model_dicts = torch.load(best_model_path, map_location=device) 
        base_model.load_state_dict(base_model_dicts)
        
        optimizer = torch.optim.Adam(params=base_model.parameters(), lr=args.lr)  # , weight_decay=0.95
        cotta_model = CoTTA_attack(model=base_model,optimizer=optimizer,arg=args) # from CoTTA to CoTTA_attack;  CoTTA_attack_softmatch
        # cotta_model = base_model.  
        cotta_model = cotta_model.to(device) 
        if args.data_parallel: # use Data Parallelism with Multi-GPU 
            cotta_model = nn.DataParallel(cotta_model)
        
        for e in range(args.Ada_epoch): 
            # train_acc, train_f1 = eval(cotta_model,args, data_loader_train)
            # valid是同一个数据集，以保证训练; test是跨数据集，来测试跨数据集的效果；
            (test_acc, test_f1), distance_sum = eval(cotta_model,args, data_loader_test, e) #eval(cotta_model,args, data_loader_test, data_loader_tta, e)
            # if nonzero_num >=2:
            #     distance.append([nonzero_mean,test_acc,test_f1])
            distance.append([0,distance_sum,test_acc,test_f1])
            for i in range(len(distance)): # 按正常顺序设置distance
                distance[i][0] = i
            distance_sort = sorted(distance,key=lambda x:x[1], reverse=True)
            print("Distance_Sort", distance_sort)
            print('Epoch %d/%d , Accuracy: %0.3f, F1: %0.3f'
                  % (e+1, args.Ada_epoch, test_acc,  test_f1)) 
            if e == 50:
                print('The Total Epoch have been reached.') 
                # print('final_selected_best_acc:',distance[2][1],'selected_best_f1:',distance[2][2])
                # print('final_selected_best_acc:',distance[final_rank][2],'selected_best_f1:',distance[final_rank][3])
                # print('the selcted epoch:',stop_epoch+1,'the selected acc:',distance[stop_epoch][2],'the selected f1:',distance[stop_epoch][3])
        # print('Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f' % best_stat)

if __name__ == "__main__": 
    args = set_arg()
    set_seeds(args.seed)
    print("Seed number in Adapt:", args.seed)
    cotta(args)