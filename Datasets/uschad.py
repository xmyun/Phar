import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

FREQUENCY = 100
SAMPLE_WINDOW = 20

COLUMNS = [
    'acc_x, w/ unit g (gravity)',
    'acc_y, w/ unit g',
    'acc_z, w/ unit g',
    'gyro_x, w/ unit dps (degrees per second)',
    'gyro_y, w/ unit dps',
    'gyro_z, w/ unit dps'
]

class USCDataset():
    """ A class for Opportunity dataset structure inculding paths to each subject and experiment file

        Attributes:
        -----------
        root_dir : str
            Path to the root directory of the dataset
        datafiles_dict : dict
            Dictionary for storing paths to each user and experiment
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.datafiles_dict = self.get_datafiles_dict()

    def get_datafiles_dict(self):
        """ Get dictionary with all subjects with corresponding raw datasets
        """
        filenames = os.listdir(self.root_dir)
        subject_folders =  [file_ for file_ in filenames if 'Subject' in file_]
        subject_folders_paths = [os.path.join(self.root_dir, file_) for file_ in subject_folders]
        
        datafiles_dict = {}
        for i, folder in enumerate(subject_folders_paths):
            tmp_subject_files = sorted(os.listdir(folder))
            tmp_activities = set([file_.split('t')[0] for file_ in tmp_subject_files]) 
            datafiles_dict[subject_folders[i]] = {}
            for act in tmp_activities:
                tmp_exps = set(['t' + file_.split('.')[0].split('t')[1] for file_ in tmp_subject_files if act in file_]) 
                datafiles_dict[subject_folders[i]][act] = {}
                for exp in tmp_exps:
                    datafiles_dict[subject_folders[i]][act][exp] = [os.path.join(folder, file_) for file_ in tmp_subject_files if act in file_ and exp in file_ and len(file_.split('t')[0]) == len(act)][0]
        return datafiles_dict
    
    def get_file(self, user_name, activity, experiment):
        """ Get file for a subject
        """
        return self.datafiles_dict[user_name][activity][experiment]

class USCInstance():
    def __init__(self, data_path):
        self.data_path = data_path
        self.user_id, self.exp_id, self.label = self.parse_userexplabel()
        self.data, self.labels_col = self.read_data()

    def parse_userexplabel(self):
        subject = self.data_path.split('/')[-2]
        filename = os.path.basename(self.data_path)
        activity = int(filename.split('t')[0][1:]) - 1
        exp = 't' + filename.split('t')[1]
        return subject, exp, activity

    def read_data(self):
        mat = loadmat(self.data_path)
        data = pd.DataFrame(mat['sensor_readings'])
        data.columns = COLUMNS
        label = int(mat['activity_number'][0]) - 1 
        labels = pd.DataFrame([label] * data.shape[0])
        labels.columns = ['label']
        return data, labels

def nested_dict_len(nested_dict):
    c = 0
    for key in nested_dict:
        if type(nested_dict[key]) is dict:
            c += nested_dict_len(nested_dict[key])
        else: 
            c += 1
    return c

def down_sample(data, window_target):
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

def test():
    data = 'D:/csshar_tfa-main/csshar_tfa-main/data/USC-HAD/'
    test_dataset = USCDataset(data)
    print(test_dataset.datafiles_dict)
    print(nested_dict_len(test_dataset.datafiles_dict))
    seq_len=120
    num=0
    for subject_idx in range(1,15): #15
        subject_str= 'Subject'+str(subject_idx)
        for activity_idx in range(1,13): #13
            activity_str= 'a'+str(activity_idx)
            for trial_idx in range(1,6): #6
                trial_str= 't'+ str(trial_idx)
                data_path = test_dataset.datafiles_dict[subject_str][activity_str][trial_str]
                print(data_path)
                test_instance = USCInstance(data_path)
                print("test_instance shape:")
                print(test_instance.data.values.shape)
                print(test_instance.labels_col.shape)
                print(test_instance.data.index[-1])
                num=num+test_instance.data.index[-1]+1

                # down-sampling
                sensor_down = down_sample(test_instance.data.values, FREQUENCY)
                if sensor_down.shape[0] >= seq_len:
                    sensor_down = sensor_down[:sensor_down.shape[0] // seq_len * seq_len, :]
                    sensor_down = sensor_down.reshape(sensor_down.shape[0] // seq_len, seq_len, sensor_down.shape[1])

                if subject_idx==1 and activity_idx==1 and trial_idx==1:
                    data_all= sensor_down
                    label_act=np.full_like(sensor_down[:,:,0],activity_idx-1)
                    label_user=np.full_like(sensor_down[:,:,0],subject_idx-1)
                    label_domain=np.full_like(sensor_down[:,:,0],4)
                else:
                    data_all= np.concatenate((data_all,sensor_down), axis=0)
                    label_act= np.concatenate((label_act,np.full_like(sensor_down[:,:,0],activity_idx-1)), axis=0)
                    label_user= np.concatenate((label_user,np.full_like(sensor_down[:,:,0],subject_idx-1)), axis=0)
                    label_domain= np.concatenate((label_domain,np.full_like(sensor_down[:,:,0],4)), axis=0)
    label_act =np.expand_dims(label_act, axis=-1)
    label_user = np.expand_dims(label_user, axis=-1)
    label_domain = np.expand_dims(label_domain, axis=-1)
    label_all = np.concatenate((label_act,label_user,label_domain),axis=-1)
    print("Finish reading.")
    print(data_all.shape)
    print(label_all.shape)
    print("Sample number (for check):")
    print(num)
    data_all[:, :, 0:3] = data_all[:, :, 0:3] * 9.81
    data_all[:, :, 3:6] = data_all[:, :, 3:6] * np.pi / 180
    np.save("D:/csshar_tfa-main/csshar_tfa-main/data/data_20_120.npy", data_all)
    np.save("D:/csshar_tfa-main/csshar_tfa-main/data/label_20_120.npy", label_all)
if __name__ == '__main__':
    test()
