import os
import numpy as np
from datasetPre import *
# from config import load_dataset_config, load_dataset_label_names

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
B  = np.unique(label[:, 1])
print(B)