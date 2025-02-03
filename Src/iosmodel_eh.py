import torch
# import coremltools as ct

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import copy
from torchattacks import PGD, FGSM
from argParse import *
from datasetPre import * 

from gram import select_model
from model import *
import io
# send f wherever
# from perceiver.model.core.modules import *
# from perceiver.model.core.config import *
# from fetch_model import fetch_classifier

class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

def cotta(args):
        data_loader_train ,data_loader_valid, data_loader_test, data_loader_tta = load_dataset(args)
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
        # print("The selected model is", best_model_path) 
        
        # Using the selected model. 
        device = get_device(args.g)
        base_model_dicts = torch.load(best_model_path, map_location=device) 
        base_model.load_state_dict(base_model_dicts)
        for data in data_loader_test:
            example = data[0]
            my_values = {
                'data': data[0],
                'label': data[1],
            }
            # torch.save(my_values, "ios_test_eh.pt")
            scripted_dict = torch.jit.script(my_values)
            torch.jit.save(scripted_dict, 'ios_test_eh.pt')
            
            container = torch.jit.script(Container(my_values))
            container.save("container.pt")
            break

        # f = io.BytesIO()
        # torch.save(example, f, _use_new_zipfile_serialization=True)
        # torch.save(example, "example.pt")
        traced_model = torch.jit.trace(base_model, example)
        # coreml_model = ct.convert(
        #     traced_model,
        #     inputs=[ct.TensorType(name="input", shape=example.shape)]
        # )
        # coreml_model.save("model.mlpackage")
        traced_model._save_for_lite_interpreter("model.ptl")

if __name__ == "__main__":
    args = set_arg()
    set_seeds(args.seed)
    print("Seed number in Adapt:", args.seed)
    cotta(args)


