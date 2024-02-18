import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from sklearn.metrics import f1_score
import argparse
import sys
import json
from scipy import signal, interpolate
from scipy.stats import special_ortho_group
from torch.nn.functional import interpolate
import torch.nn.functional as F
class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError
class Preprocess4Sample_t(Pipeline):
    
    def __init__(self, seq_len, temporal=0.4, temporal_range=[0.8, 1.2]):
        super().__init__()
        self.seq_len = seq_len
        self.temporal = temporal
        self.temporal_range = temporal_range

    def __call__(self, batch_instance):
        # Assuming batch_instance is a tensor of shape (batch_size, seq_len, feature_len)
        # processed_batch = []

        # for instance in batch_instance:
        #     print(instance.shape)
        #     if instance.shape[0] == self.seq_len:
        #         processed_instance = instance
        #     elif self.temporal > 0:
        #         temporal_prob = torch.rand(1).item()
        #         if temporal_prob < self.temporal:
        #             x = torch.arange(instance.shape[0])
        #             ratio_random = torch.rand(1).item() * (self.temporal_range[1] - self.temporal_range[0]) + self.temporal_range[0]
        #             seq_len_scale = int(round(ratio_random * self.seq_len))
        #             index_rand = torch.randint(0, instance.shape[0] - seq_len_scale, (1,)).item()
        #             instance_new = torch.zeros((self.seq_len, instance.shape[1]), device=instance.device)
        #             x_new = torch.linspace(index_rand, index_rand + seq_len_scale, self.seq_len, device=instance.device)
        #             for i in range(instance.shape[1]):
        #                 # f = torch.interpolate(torch.tensor([x, instance[:, i]]), x_new, mode='linear')
        #                 f = torch.interp(x, sorted(x), instance[:, i])
        #                 instance_new[:, i] = f
        #             processed_instance = instance_new
        #         else:
        #             index_rand = torch.randint(0, instance.shape[0] - self.seq_len, (1,)).item()
        #             processed_instance = instance[index_rand:index_rand + self.seq_len, :]
        #     else:
        #         index_rand = torch.randint(0, instance.shape[0] - self.seq_len, (1,)).item()
        #         processed_instance = instance[index_rand:index_rand + self.seq_len, :]
        #     processed_batch.append(processed_instance)
        # processed_batch_tensor = torch.stack(processed_batch)
        # print(batch_instance.shape)
        processed_batch_tensor = interpolate(batch_instance.unsqueeze(0), size = (240,6), mode='bilinear')
        # print(processed_batch_tensor.shape)
        random_index = torch.randint(low = 0, high = processed_batch_tensor.shape[1] - self.seq_len, size = (1,))
        processed_batch_tensor = processed_batch_tensor.squeeze(0)
        return processed_batch_tensor[:, random_index:random_index + batch_instance.shape[1], :]
    "the 156 should match the linear layer of dcnn"

class Preprocess4Normalization_t():

    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma

    def __call__(self, instance):
        # Assuming instance is a tensor of shape (seq_len, feature_len) or (batch_size, seq_len, feature_len)
        # Truncate to the specified feature length
        instance_new = instance.clone()[..., :self.feature_len]
        
        if instance_new.shape[-1] >= 6 and self.norm_acc:
            # Normalize acceleration data
            instance_new[..., :3] = instance_new[..., :3] / self.acc_norm
        
        if instance_new.shape[-1] == 9 and self.norm_mag:
            # Normalize magnetometer data
            mag_norms = torch.norm(instance_new[..., 6:9], dim=-1, keepdim=True) + self.eps
            mag_norms = mag_norms.repeat(1, 1, 3) if mag_norms.dim() == 3 else mag_norms.repeat(1, 3)
            instance_new[..., 6:9] = (instance_new[..., 6:9] / mag_norms) * self.gamma
        
        return instance_new
    
class Preprocess4Permute_t():
    
    def __init__(self, segment_size=4):
        super().__init__()
        self.segment_size = segment_size

    def __call__(self, instance):
        # Assuming instance has shape [batch_size, 1, time_steps, features]
        batch_size, time_steps, features = instance.shape
        # Calculate the number of segments
        num_segments = time_steps // self.segment_size
        # Reshape to have segments as the first dimension
        instance = instance.view(batch_size, num_segments, self.segment_size, features)
        # Generate a random permutation of segments
        permuted_indices = torch.randperm(num_segments)
        # Apply the permutation to the segments
        instance = instance[:, permuted_indices, :]
        # Reshape back to the original shape
        instance = instance.view(batch_size, time_steps, features)
        return instance


class Preprocess4Rotation_t():
    def __init__(self, sensor_dimen=3):
        super().__init__()
        self.sensor_dimen = sensor_dimen

    def __call__(self, instance):
        return self.rotate_random(instance)

    def rotate_random(self, instance):
        # Assuming instance has shape [batch_size, 1, time_steps, features]
        batch_size, time_steps, features = instance.shape
        instance_new = instance.view(batch_size, time_steps, features // self.sensor_dimen, self.sensor_dimen)
        rotation_matrix = torch.tensor(special_ortho_group.rvs(self.sensor_dimen)).float().to(instance.device)
        instance_new = torch.einsum('bijk,kl->bijl', instance_new, rotation_matrix)
        return instance_new.view(batch_size, time_steps, features)
class Preprocess4Noise_t():
    def __init__(self, p=1.0, mu=0.0, var=0.1):
        super().__init__()
        self.p = p
        self.mu = mu
        self.var = var
    def __call__(self, instance):
        if torch.rand(1).item() < self.p:
            noise = torch.normal(mean=self.mu, std=self.var, size=instance.shape).to(instance.device)
            instance += noise
        return instance


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)

def stat_acc_f1(label, results_estimated):
    # label = np.concatenate(label, 0)
    # results_estimated = np.concatenate(results_estimated, 0)
    label_estimated = np.argmax(results_estimated, 1)
    f1 = f1_score(label, label_estimated, average='macro')
    acc = np.sum(label == label_estimated) / label.size
    return acc, f1

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)
class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError

class IMUDataset(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, labels, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        instance = self.data[index]
        # print(len(self.data))
        for proc in self.pipeline:
            instance = proc(instance)
        return torch.from_numpy(instance).float(), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)
def narray_to_tensor(narray_list):
    for i in range(len(narray_list)):
        item = narray_list[i]
        if item.dtype == np.float32 or item.dtype == np.float64:
            narray_list[i] = torch.from_numpy(item).float()
        else:
            narray_list[i] = torch.from_numpy(item).long()
    return narray_list
class IMUTTADataset(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, labels, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        instance = self.data[index]
        # print(len(self.data))
        for proc in self.pipeline:
            instance = proc(instance)
        result=[]
        if isinstance(instance, tuple):
            result += list(instance)
        else:
            result += [instance]
        result.append(np.array(self.labels[index]))
        return narray_to_tensor(list(result))

    def __len__(self):
        return len(self.data)

class FFTDataset(Dataset):
    def __init__(self, data, labels, mode=0, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels
        self.mode = mode

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        seq = self.preprocess(instance)
        return torch.from_numpy(seq), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)

    def preprocess(self, instance):
        f = np.fft.fft(instance, axis=0, n=10)
        mag = np.abs(f)
        phase = np.angle(f)
        return np.concatenate([mag, phase], axis=0).astype(np.float32)
def load_dataset(args):
    #DATASET_PATH = r'F:\Dataset_Mobility\UCI HAR Dataset Raw\RawData'
    #only for uci and shoaib right now
    # def preprocess(path, path_save, version, raw_sr=50, target_sr=20, seq_len=20):
    if ("uci" in args.dataset):
        data, label = load_uci(args)
    elif ("hhar" in args.dataset):
        data, label = load_hhar(args)
    elif ("shoaib" in args.dataset):
        data, label = load_shoaib(args)
    elif ("motion" in args.dataset):
        data, label = load_motion(args)
    data_train, label_train, data_vali, label_vali, data_test, label_test = partition_and_reshape(data, label)
    #if you need to train with part of labeled data, you can use the following code:
    # data_train, label_train, _, _  = prepare_simple_dataset(data_train, label_train,training_rate=1)
    # pipeline = [Preprocess4Normalization(args.input)]
    pipeline=[]
    pipeline_tta = [Preprocess4Normalization(args.input),Preprocess4Sample(args.seq_len, temporal=0.4)
            , Preprocess4Rotation(), Preprocess4Noise(), Preprocess4Permute()]
    if args.model != 'deepsense':
        data_set_train = IMUDataset(data_train, label_train, pipeline=pipeline)
        data_set_test = IMUDataset(data_test, label_test, pipeline=pipeline)
        data_set_vali = IMUDataset(data_vali, label_vali, pipeline=pipeline)
        data_set_tta = IMUTTADataset(data_test, label_train, pipeline=pipeline_tta) # label_test
    else:
        data_set_train = FFTDataset(data_train, label_train, pipeline=pipeline)
        data_set_test = FFTDataset(data_test, label_test, pipeline=pipeline)
        data_set_vali = FFTDataset(data_vali, label_vali, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=args.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=args.batch_size)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=args.batch_size)
    data_loader_tta = DataLoader(data_set_tta, shuffle=False, batch_size=args.batch_size)
    return data_loader_train, data_loader_vali, data_loader_test, data_loader_tta
def down_sample_uci(data, window_sample, start, end):
    result = []
    if window_sample.is_integer():
        window = int(window_sample)
        for i in range(start, end - window, window):
            slice = data[i: i + window, :]
            result.append(np.mean(slice, 0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = int(start)
        while int(start) <= i + window + 1 < int(end):
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                slice = data[i: i + window + 1, :]
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window, start, end))
                result.append(np.mean(slice, 0))
                i += window + 1
            else:
                slice = data[i: i + window, :]
                result.append(np.mean(slice, 0))
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window +  1, start, end))
                i += window
    return np.array(result)
def load_uci(args,raw_sr=50, target_sr=20, seq_len=120,path=""):
    # st = os.path.join((path, 'labels.txt'), delimiter=' ')
    # print(st)
    labels = np.loadtxt("/mnt/home/xuemeng/ttaIMU/cotta_imu/dataset/uci/RawData/labels.txt")
    data = []
    label = []
    window_sample = raw_sr / target_sr
    for root, dirs, files in os.walk("/mnt/home/xuemeng/ttaIMU/cotta_imu/dataset/uci/RawData"):
        for name in files:
            if name.startswith('acc'):
                tags = name.split('.')[0].split('_')
                exp_num = int(tags[1][-2:])
                exp_user = int(tags[2][-2:])
                label_index = (labels[:, 0] == exp_num) & (labels[:, 1] == exp_user)
                label_stat = labels[label_index, :]
                for i in range(label_stat.shape[0]):
                    index_start = label_stat[i, 3]
                    index_end = label_stat[i, 4]
                    exp_data_acc = np.loadtxt(os.path.join(root, name), delimiter=' ') * 9.80665
                    exp_data_gyro = np.loadtxt(os.path.join(root, 'gyro' + name[3:]), delimiter=' ')
                    exp_data = down_sample_uci(np.concatenate([exp_data_acc, exp_data_gyro], 1), window_sample, index_start, index_end)
                    if exp_data.shape[0] > seq_len and label_stat[i, 2] <= 6:
                        exp_data = exp_data[:exp_data.shape[0] // seq_len * seq_len, :]
                        exp_data = exp_data.reshape(exp_data.shape[0] // seq_len, seq_len, exp_data.shape[1])
                        exp_label = np.ones((exp_data.shape[0], exp_data.shape[1], 1))
                        exp_label = np.concatenate([exp_label * label_stat[i, 2], exp_label * label_stat[i, 1]], 2)
                        data.append(exp_data)
                        label.append(exp_label)
    data = np.concatenate(data, 0)
    label = np.concatenate(label, 0)
    label[:, :, 0] = label[:, :, 0] - np.min(label[:, :, 0])
    label[:, :, 1] = label[:, :, 1] - np.min(label[:, :, 1])
    print("data:",data.shape)
    return data, label
ACT_LABELS = ["walking", "sitting", "standing", "jogging", "biking", "upstairs" , "downstairs"]
SAMPLE_WINDOW = 20
def label_name_to_index(label_names):
    label_index = np.zeros(label_names.size)
    for i in range(len(ACT_LABELS)):
        ind = label_names == ACT_LABELS[i]
        # print(np.sum(ind))
        label_index[ind] = i
    return label_index


def down_sample_shoaib(data, window_target):
    window_sample = window_target * 1.0 / SAMPLE_WINDOW
    result = []
    if window_sample.is_integer():
        window = int(window_sample)
        for i in range(0, len(data), window):
            slice = data[i: i + window, :]
            result.append(np.mean(slice, 0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = 0
        while 0 <= i + window + 1 <= data.shape[0]:
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                slice = data[i: i + window + 1, :]
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window, start, end))
                result.append(np.mean(slice, 0))
                i += window + 1
            else:
                slice = data[i: i + window, :]
                result.append(np.mean(slice, 0))
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window +  1, start, end))
                i += window
    return np.array(result)


def load_shoaib(args, target_window=50, seq_len=120, position_num=5):
    data = []
    label = []
    for root, dirs, files in os.walk('/mnt/home/xuemeng/ttaIMU/cotta_imu/dataset/shoaib'):
        for f in range(len(files)):
            if 'Participant' in files[f]:
                exp = pd.read_csv(os.path.join(root, files[f]), skiprows=1)
                labels_activity = exp.iloc[:, -1].to_numpy()
                labels_activity = label_name_to_index(labels_activity)
                for a in range(len(ACT_LABELS)):
                    exp_act = exp.iloc[labels_activity == a, :]
                    for i in range(position_num):
                        index = np.array([1, 2, 3, 7, 8, 9, 10, 11, 12]) + i * 14
                        exp_pos = exp_act.iloc[:, index].to_numpy(dtype=np.float32)
                        print("User-%s, activity-%s, position-%d: num-%d" % (files[f], ACT_LABELS[a], i, exp_pos.shape[0]))
                        if exp_pos.shape[0] > 0:
                            exp_pos_down = down_sample_shoaib(exp_pos, target_window)
                            sensor_down = exp_pos_down[:exp_pos_down.shape[0] // seq_len * seq_len, :]
                            sensor_down = sensor_down.reshape(sensor_down.shape[0] // seq_len, seq_len, sensor_down.shape[1])
                            sensor_label = np.multiply(np.ones((sensor_down.shape[0], sensor_down.shape[1], 1)),
                                                       np.array([a, i, f]).reshape(1, 3))
                            data.append(sensor_down)
                            label.append(sensor_label)
    data = np.concatenate(data, 0)
    label = np.concatenate(label, 0)
    return data, label





def extract_sensor(data, time_index, time_tag, window_time):
    index = time_index
    while index < len(data) and abs(data.iloc[index]['Creation_Time'] - time_tag) < window_time:
        index += 1
    if index == time_index:
        return None, index
    else:
        data_slice = data.iloc[time_index:index]
        if data_slice['User'].unique().size > 1 or data_slice['gt'].unique().size > 1:
            return None, index
        else:
            data_sensor = data_slice[['x', 'y', 'z']].to_numpy()
            sensor = np.mean(data_sensor, axis=0)
            label = data_slice[['User', 'Model', 'gt']].iloc[0].values
            return np.concatenate([sensor, label]), index

def transform_to_index(label, print_label=False):
    labels_unique = np.unique(label)
    if print_label:
        print(labels_unique)
    for i in range(labels_unique.size):
        label[label == labels_unique[i]] = i

def separate_data_label(data_raw):
    labels = data_raw[:, :, -3:].astype(np.str)
    transform_to_index(labels[:, :, 0])
    transform_to_index(labels[:, :, 1], print_label=True)
    transform_to_index(labels[:, :, 2], print_label=True)
    data = data_raw[:, :, :6].astype(np.float)
    labels = labels.astype(np.float)
    return data, labels


# 'Index', 'Arrival_Time', 'Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'Device', 'gt'
def load_hhar(args, window_time=50, seq_len=40, jump=0):
    accs = pd.read_csv(args.path + '\Phones_accelerometer.csv')
    gyros = pd.read_csv(args.path + '\Phones_gyroscope.csv') #, nrows=200000
    time_tag = min(accs.iloc[0, 2], gyros.iloc[0, 2])
    time_index = [0, 0] # acc, gyro
    window_num = 0
    data = []
    data_temp = []
    while time_index[0] < len(accs) and time_index[1] < len(gyros):
        acc, time_index_new_acc = extract_sensor(accs, time_index[0], time_tag, window_time=window_time * pow(10, 6))
        gyro, time_index_new_gyro = extract_sensor(gyros, time_index[1], time_tag, window_time=window_time * pow(10, 6))
        time_index = [time_index_new_acc, time_index_new_gyro]
        if acc is not None and gyro is not None and np.all(acc[-3:] == gyro[-3:]):
            time_tag += window_time * pow(10, 6)
            window_num += 1
            data_temp.append(np.concatenate([acc[:-3], gyro[:-3], acc[-3:]]))
            if window_num == seq_len:
                data.append(np.array(data_temp))
                if jump == 0:
                    data_temp.clear()
                    window_num = 0
                else:
                    data_temp = data_temp[-jump:]
                    window_num -= jump
        else:
            if window_num > 0:
                data_temp.clear()
                window_num = 0
            if time_index[0] < len(accs) and time_index[1] < len(gyros):
                time_tag = min(accs.iloc[time_index[0], 2], gyros.iloc[time_index[1], 2])
            else:
                break
    data_raw = np.array(data)
    data_new, label_new = separate_data_label(data_raw)
    return data_new, label_new




ACTIVITY_NAMES = ["dws", "ups", "sit", "std", "wlk", "jog"]


def label_activity(name):
    for i in range(len(ACTIVITY_NAMES)):
        if name.startswith(ACTIVITY_NAMES[i]):
            return i


def label_user(name):
    temp = name.split(".")[0]
    id = int(temp.split("_")[1])
    return id - 1


def down_sample_motion(data, window_target):
    window_sample = window_target * 1.0 / SAMPLE_WINDOW
    result = []
    if window_sample.is_integer():
        window = int(window_sample)
        for i in range(0, len(data), window):
            slice = data[i: i + window, :]
            result.append(np.mean(slice, 0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = 0
        while 0 <= i + window + 1 < data.shape[0]:
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                slice = data[i: i + window + 1, :]
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window, start, end))
                result.append(np.mean(slice, 0))
                i += window + 1
            else:
                slice = data[i: i + window, :]
                result.append(np.mean(slice, 0))
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window +  1, start, end))
                i += window
    return np.array(result)


def load_sensor_data(path, seq_len, target_window):
    data = []
    label = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            path_act = os.path.join(root, dir)
            label_act = label_activity(dir)
            for root_exp, dirs_exp, files_exp in os.walk(path_act):
                for name in files_exp:
                    path_exp = os.path.join(root_exp, name)
                    label_u = label_user(name)
                    sensor = np.loadtxt(path_exp, skiprows=1, delimiter=',')
                    sensor_down = down_sample(sensor[:, 1:], target_window)
                    if sensor_down.shape[0] > seq_len:
                        sensor_down = sensor_down[:sensor_down.shape[0] // seq_len * seq_len, :]
                        sensor_down = sensor_down.reshape(sensor_down.shape[0] // seq_len, seq_len, sensor_down.shape[1])
                        sensor_label = np.ones((sensor_down.shape[0], sensor_down.shape[1], 1))
                        sensor_label = np.concatenate([sensor_label * label_act, sensor_label * label_u], 2)
                        data.append(sensor_down)
                        label.append(sensor_label)
    return data, label


def load_motion(args,target_window=50, seq_len=20):
    data_acc, label_acc = load_sensor_data(os.path.join(args.path, 'Accelerometer'), seq_len, target_window)
    data_gyro, label_gyro = load_sensor_data(os.path.join(args.path, 'Gyroscope'), seq_len, target_window)
    data = []
    label = []
    for i in range(len(data_acc)):
        len_min = min(data_acc[i].shape[0], data_gyro[i].shape[0])
        data.append(np.concatenate([data_acc[i][:len_min] * 9.8, data_gyro[i][:len_min]], 2))
        label.append(label_acc[i][:len_min, :, :])
    data = np.concatenate(data, 0)
    label = np.concatenate(label, 0)
    return data, label

def reshape_data(data, merge):
    if merge == 0:
        return data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    else:
        return data.reshape(data.shape[0] * data.shape[1] // merge, merge, data.shape[2])
def reshape_label(label, merge):
    if merge == 0:
        return label.reshape(label.shape[0] * label.shape[1])
    else:
        return label.reshape(label.shape[0] * label.shape[1] // merge, merge)
def merge_dataset(data, label, mode='all'):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        if mode == 'all':
            temp_label = np.unique(label[i])
            if temp_label.size == 1:
                index[i] = True
                label_new.append(label[i, 0])
        elif mode == 'any':
            index[i] = True
            if np.any(label[i] > 0):
                temp_label = np.unique(label[i])
                if temp_label.size == 1:
                    label_new.append(temp_label[0])
                else:
                    label_new.append(temp_label[1])
            else:
                label_new.append(0)
        else:
            index[i] = ~index[i]
            label_new.append(label[i, 0])
    # print('Before Merge: %d, After Merge: %d' % (data.shape[0], np.sum(index)))
    return data[index], np.array(label_new)
def partition_and_reshape(data, labels, label_index=0, training_rate=0.8, vali_rate=0.1, change_shape=True
                          , merge=20, merge_mode='all', shuffle=False):
    arr = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    vali_num = int(data.shape[0] * vali_rate)
    data_train = data[:train_num, ...]
    data_vali = data[train_num:train_num+vali_num, ...]
    data_test = data[train_num+vali_num:, ...]
    t = np.min(labels[:, :, label_index])
    label_train = labels[:train_num, ..., label_index] - t
    label_vali = labels[train_num:train_num+vali_num, ..., label_index] - t
    label_test = labels[train_num+vali_num:, ..., label_index] - t
    if change_shape:
        data_train = reshape_data(data_train, merge)
        data_vali = reshape_data(data_vali, merge)
        data_test = reshape_data(data_test, merge)
        label_train = reshape_label(label_train, merge)
        label_vali = reshape_label(label_vali, merge)
        label_test = reshape_label(label_test, merge)
    if change_shape and merge != 0:
        data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
        data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)
        data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
    print('Train Size: %d, Vali Size: %d, Test Size: %d' % (label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
    return data_train, label_train, data_vali, label_vali, data_test, label_test
def prepare_simple_dataset(data, labels, training_rate=0.2):
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    data_train = data[:train_num, ...]
    data_test = data[train_num:, ...]
    t = np.min(labels)
    label_train = labels[:train_num] - t
    label_test = labels[train_num:] - t
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    print('Label Size: %d, Unlabel Size: %d. Label Distribution: %s'
          % (label_train.shape[0], label_test.shape[0], ', '.join(str(e) for e in label_num)))
    return data_train, label_train, data_test, label_test
def set_arg():
    parser = argparse.ArgumentParser(description='tta_imu training')
    parser.add_argument('--model', type=str, help='The model you want to use')
    parser.add_argument('--dataset', type=str, help='Dataset name', choices=['hhar_20_120', 'motion_20_120', 'uci_20_120', 'shoaib_20_120'])
    parser.add_argument('--source_dataset', type=str, default="uci_20_120", help='source model name', choices=['hhar_20_120', 'motion_20_120', 'uci_20_120', 'shoaib_20_120'])
    args = parser.parse_args()
    json_data_path = '/mnt/home/xuemeng/ttaIMU/IMU_Hete/config/dataset.json'
    json_model_path = '/mnt/home/xuemeng/ttaIMU/IMU_Hete/config/model.json'
    json_train_path = '/mnt/home/xuemeng/ttaIMU/IMU_Hete/config/train.json'
    json_mask_path = '/mnt/home/xuemeng/ttaIMU/IMU_Hete/config/mask.json'
    with open(json_model_path) as f:
        json_data = json.load(f)
        model_config = json_data.get(args.model)
    with open(json_data_path) as f:
        json_data = json.load(f)
        data_config = json_data.get(args.dataset)
    with open(json_train_path) as f:
        train_config = json.load(f)
    with open(json_mask_path) as f:
        mask_config = json.load(f)
    args.__dict__.update(train_config)
    args.__dict__.update(data_config)
    args.__dict__.update(model_config)
    args.__dict__.update(mask_config)
    return args

def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)

class Preprocess4Normalization(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma

    def __call__(self, instance):
        # print(instance.shape)
        instance_new = instance.copy()[:, :self.feature_len]
        # if instance_new.shape[1] >= 6 and self.norm_acc:
        #     instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        if instance_new.shape[1] == 9 and self.norm_mag:
            mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
            mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
            instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        return instance_new

class Preprocess4Mask():

    def __init__(self, mask_cfg, full_sequence=False):
        self.mask_ratio = mask_cfg.mask_ratio  # masking probability
        self.mask_alpha = mask_cfg.mask_alpha
        self.max_gram = mask_cfg.max_gram
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob
        self.full_sequence = full_sequence

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def __call__(self, instance):
        shape = instance.shape

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        mask_pos = span_mask(shape[0], self.max_gram,  goal_num_predict=n_pred)

        instance_mask = instance.copy()
        if np.random.rand() < self.mask_prob:
            instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
        elif np.random.rand() < self.mask_prob + self.replace_prob:
            instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
        seq = instance[mask_pos, :]
        if self.full_sequence:
            return instance, instance_mask, np.array(mask_pos), np.array(seq)
        else:
            return instance_mask, np.array(mask_pos), np.array(seq)
class Preprocess4Rotation():

    def __init__(self, sensor_dimen=3):
        super().__init__()
        self.sensor_dimen = sensor_dimen

    def __call__(self, instance):
        return self.rotate_random(instance)

    def rotate_random(self, instance):
        # print(instance.shape)
        instance_new = instance.reshape(instance.shape[0], instance.shape[1] // self.sensor_dimen, self.sensor_dimen)
        rotation_matrix = special_ortho_group.rvs(self.sensor_dimen)
        for i in range(instance_new.shape[1]):
            instance_new[:, i, :] = np.dot(instance_new[:, i, :], rotation_matrix)
        return instance_new.reshape(instance.shape[0], instance.shape[1])
class Preprocess4Sample():
    
    def __init__(self, seq_len, temporal=0.4, temporal_range=[0.8, 1.2]):
        super().__init__()
        self.seq_len = seq_len
        self.temporal = temporal
        self.temporal_range = temporal_range

    def __call__(self, instance):
        if instance.shape[0] == self.seq_len:
            return instance
        if self.temporal > 0:
            temporal_prob = np.random.random()
            if temporal_prob < self.temporal:
                x = np.arange(instance.shape[0])
                ratio_random = np.random.random() * (self.temporal_range[1] - self.temporal_range[0]) + self.temporal_range[0]
                seq_len_scale = int(np.round(ratio_random * self.seq_len))
                index_rand = np.random.randint(0, high=instance.shape[0] - seq_len_scale)
                instance_new = np.zeros((self.seq_len, instance.shape[1]))
                for i in range(instance.shape[1]):
                    f = interpolate.interp1d(x, instance[:, i], kind='linear')
                    x_new = index_rand + np.linspace(0, seq_len_scale, self.seq_len)
                    instance_new[:, i] = f(x_new)
                return instance_new
        index_rand = np.random.randint(0, high=instance.shape[0] - self.seq_len)
        return instance[index_rand:index_rand + self.seq_len, :]
    
class Preprocess4Permute(Pipeline):
    
    def __init__(self, segment_size=4):
        super().__init__()
        self.segment_size = segment_size

    def __call__(self, instance):
        original_shape = instance.shape
        instance = instance.reshape(self.segment_size, instance.shape[0]//self.segment_size, -1)
        order = np.random.permutation(self.segment_size)
        instance = instance[order, :, :]
        return instance.reshape(original_shape)
    
class Preprocess4Noise(Pipeline):
    
    def __init__(self, p=1.0, mu=0.0, var=0.1):
        super().__init__()
        self.p = p
        self.mu = mu
        self.var = var

    def __call__(self, instance):
        if np.random.random() < self.p:
            instance += np.random.normal(self.mu, self.var, instance.shape)
        return instance
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

# def kmmd_dist(x1, x2):
#     X_total = torch.cat([x1,x2],0)
#     Gram_matrix = gaussian_kernel(X_total,X_total,kernel_mul=2.0, kernel_num=2, fix_sigma=0, mean_sigma=0)
#     n = int(x1.shape[0])
#     m = int(x2.shape[0])
#     print(n,m)
#     print("gram",Gram_matrix)
#     x1x1 = Gram_matrix[:n, :n]
#     x2x2 = Gram_matrix[n:, n:]
#     x1x2 = Gram_matrix[:n, n:]
#     # x2x1 = Gram_matrix[n:, :n]  # Gram_matrix is symmetric
#     diff = torch.mean(x1x1) + torch.mean(x2x2) - 2 * torch.mean(x1x2)
#     diff = (m*n)/(m+n)*diff
#     return diff.to(torch.device('cpu')).numpy()

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
