import json
import argparse
import sys
# json_data_path = 'config/'+ args.dataset + '.json'
parser = argparse.ArgumentParser(description='tta_imu training')
parser.add_argument('model', type=str, help='The model you want to use')
parser.add_argument('dataset', type=str, help='Dataset name', choices=['hhar_20_120', 'motion_20_120', 'uci_20_120', 'shoaib_20_120'])
args = parser.parse_args()
json_data_path = 'config/dataset.json'
json_model_path = 'config/model.json'
json_train_path = 'config/train.json'
with open(json_model_path) as f:
    json_data = json.load(f)
    model_config = json_data.get(args.model)
with open(json_data_path) as f:
    json_data = json.load(f)
    data_config = json_data.get(args.dataset)
with open(json_train_path) as f:
    train_config = json.load(f)
# print(model_config)
# print(data_config)
# for key, value in model_config.items():
#     # parser.add_argument(key)
#     # args.key = value
#     args.append(f'--{key}')
#     # args.append(value)
# for key, value in data_config.items():
#     # parser.add_argument(key)
#     # parser.key = value
#     # args.key = value
#     args.append(f'--{key}')
# for key, value in train_config.items():
#     # parser.add_argument(key)
#     # parser.key = value
#     # args.key = value
#     # print(key,args.key)
#     args.append(f'--{key}')
# args = parser.parse_args()
args.__dict__.update(train_config)
args.__dict__.update(data_config)
args.__dict__.update(model_config)
print(args.seed)