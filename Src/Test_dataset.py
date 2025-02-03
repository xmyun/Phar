import os
import numpy as np
from datasetPre import *
from config import load_dataset_config, load_dataset_label_names

np.set_printoptions(threshold=np.inf)

path_label = os.path.join("/label_" + "20_120" + ".npy") 
path_data = os.path.join("/data_" + "20_120" + ".npy") 
path_label = "/mnt/home/xuemeng/ttaIMU/Artifact-UniHAR/dataset/merge" + path_label
path_data = "/mnt/home/xuemeng/ttaIMU/Artifact-UniHAR/dataset/merge" + path_data
# path_label = "/mnt/home/xuemeng/ttaIMU/cotta_imu/Datasets/Merge/Read2Py" + path_label
print(path_label)
label = np.load(path_label).astype(np.float32)
data = np.load(path_data).astype(np.float32)
user_label_index = 1

# print(np.unique(label[:, 1])) 
# print(np.unique(label[:,0]).size) #label_user, label_domain, label_act_new
# print(np.unique(label[:,:,2])) # label dataset;
# print(np.unique(label[:,:,user_label_index])) # label_user
# print(np.unique(label[:,:,0])) # label_act
# print(label[:,0,2])


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

# domain 1_9: 0-8; 2_30: 9-38; 3_24: 39-62; 4_10: 63-72 ; 0_73: 0-72
UserN= check_domain_user_num(label, 2, label_user_index=1, label_domain_index=2, domain_num=4) #domian=0, 表示读取全部的数据；
print(UserN) 
# users_total = np.unique(np.unique(label[:,:,user_label_index]))
# print(users_total)
User_list= np.array([64,66,67,70,72]) # User list. 

datas, labels = filter_domain_list(data, label, 4)  # Extract data in dataset/domain 1;
datas, labels = filter_dataset(datas, labels)
datas, labels = filter_label_max(datas, labels, 5) # Cutoff four activity. 
dataset = Extractor_Users(datas, labels, User_list, user_label_index=1)

data_set = [item[0] for item in dataset]
label_set = [item[1][:, 0] for item in dataset]
print("selected users", len(label_set))
result = [[], [], [], [], [], []]
# For each user, confiure train/vali/test. 
for i in range(len(data_set)): 
    data_train_temp, data_test, label_train_temp, label_test \
         = train_test_split(data_set[i], label_set[i], train_size=0.7) #, random_state=seed
    training_size_new = 0.7 / (1 - 0.1) 
    print(i) 
    data_train, data_vali, label_train, label_vali \
        = train_test_split(data_train_temp, label_train_temp, train_size=training_size_new) #, random_state=seed
    [data_train, data_vali, data_test] = reshape_data([data_train, data_vali, data_test], 0)
    [label_train, label_vali, label_test] = reshape_label([label_train, label_vali, label_test],
                                                                    data.shape[1], 0)
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
    
data_train, label_train, data_vali, label_vali, data_test, label_test =result
print(label_test)
print(data_train.shape)

# Write the code for Preprocessing pipeline, then conduct dataload. 


# # Read dataset info. 
# ArAns = load_dataset_config("uci", "20_120")
# label_names, label_num =load_dataset_label_names(ArAns,0,0)
# print(label_names,label_num )
