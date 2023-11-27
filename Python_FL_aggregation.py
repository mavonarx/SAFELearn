import numpy as np
import pandas as pd
import torch
import PPMI_prediction_NN
import torch.nn as nn


###############################################################################

PROJECT = "PPMI"
INPUT_DATA_PATH = f"input_data/{PROJECT}/PPMI_cleaned_altered.csv"
MODEL_PATH= f"model/{PROJECT}/"
GLOBAL_MODEL_PATH = f"{MODEL_PATH}/GlobalModel.txt"

###############################################################################



class PPMIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(12,20)
        self.linear2 = nn.Linear(20,15)
        self.linear3 = nn.Linear(15,12)
        self.linear4 = nn.Linear(12,4)
        self.linear5 = nn.Linear(4,3)
        self.act_fn = nn.SiLU()
        self.sigm = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        x = self.act_fn(x)
        x = self.linear4(x)
        x = self.act_fn(x)
        x = self.linear5(x)
        return x
    
    

global_model = PPMIModel()
global_model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
delta_wk_h_np = np.loadtxt('model/PPMI/Delta_{i}.txt')
vec = torch.tensor(delta_wk_h_np)

summed_deltas = vec[1:]
summed_h = vec[0]

for i in range(1, PPMI_prediction_NN.clients):
    #delta_file = f'Delta_{i:03d}.txt'
    delta_wk_h_np = np.loadtxt('model/PPMI/Delta_{i}.txt')
    vec = torch.tensor(delta_wk_h_np)
    summed_h += vec[0]
    summed_deltas += vec[1:]
    

result = global_model - (summed_deltas / summed_h)

torch.save(global_model.state_dict(), GLOBAL_MODEL_PATH)