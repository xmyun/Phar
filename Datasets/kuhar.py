import numpy as np
import pandas as pd
import os

SAMPLE_WINDOW=20
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

CLASS_LABELS = np.array(
    [
        "Stand", #0
        "Sit", #1
        "Talk-sit", #2
        "Talk-stand", #3
        "Stand-sit", #4
        "Lay", #5
        "Lay-stand", #6
        "Pick", #7
        "Jump", #8
        "Push-up", #9
        "Sit-up", #10
        "Walk", #11
        "Walk-backwards", #12
        "Walk-circle", #13
        "Run", #14
        "Stair-up", #15
        "Stair-down", #16
        "Table-tennis", #17
    ]
)
num=0
for i in range(18): #18
    if i!=0 and i!=1 and i!=5 and i!=11 and i!=13 and i!=15 and i!=16:
        continue
    folder_path ="./dataset/2.Trimmed_interpolated_data/"+str(i)+"."+CLASS_LABELS[i]+"/"
    print(folder_path)
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    print(file_list)
    for file_name in sorted(file_list):
        csv_path = os.path.join(folder_path, file_name)
        num=num+1
        # load csv file
        df = pd.read_csv(csv_path, header=None)
        acc_signals = df.values[:, 1:4]
        gyro_signals = df.values[:, 5:8]
        acc_signals = np.array(acc_signals, dtype=np.float32)
        gyro_signals = np.array(gyro_signals, dtype=np.float32)
        signals = np.concatenate((acc_signals, gyro_signals), axis=-1)
        # down-sampling
        FREQUENCY=100
        seq_len=120
        #print(signals.shape)
        sensor_down = down_sample(signals, FREQUENCY)
        if sensor_down.shape[0] >= seq_len:
            sensor_down = sensor_down[:sensor_down.shape[0] // seq_len * seq_len, :]
            sensor_down = sensor_down.reshape(sensor_down.shape[0] // seq_len, seq_len, sensor_down.shape[1])
            #print(sensor_down.shape)
            print(sensor_down[0,0,:])
            sensor_down[:, :, 0:3] = sensor_down[:, :, 0:3] * 9.81
            if num==1:
                data_all = sensor_down
                label_act = np.full_like(sensor_down[:,:,0], i)
                label_user = np.full_like(sensor_down[:,:,0], int(file_name[0:4])-1001)
                label_domain = np.full_like(sensor_down[:,:,0], 5)
            else:
                data_all = np.concatenate((data_all, sensor_down), axis=0)
                label_act = np.concatenate((label_act, np.full_like(sensor_down[:,:,0], i)), axis=0)
                user_idx=int(file_name[0:4])-1001
                if  user_idx==100:
                    user_idx=79
                else:
                    if user_idx >58:
                        user_idx=user_idx - 1
                    if user_idx >10:
                        user_idx = user_idx - 1
                label_user = np.concatenate((label_user, np.full_like(sensor_down[:,:,0], user_idx)), axis=0)
                label_domain = np.concatenate((label_domain, np.full_like(sensor_down[:,:,0], 5)), axis=0)
label_act = np.expand_dims(label_act, axis=-1)
label_user = np.expand_dims(label_user, axis=-1)
label_domain = np.expand_dims(label_domain, axis=-1)
label_all = np.concatenate((label_act,label_user,label_domain),axis=-1)
print(data_all.shape)
print(label_all.shape)
print(np.unique(label_act))
print(np.unique(label_user))
print(np.unique(label_user).shape)
print(np.max(data_all))
np.save("./dataset/data_20_120.npy",data_all)
np.save("./dataset/label_20_120.npy",label_all)