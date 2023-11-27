import numpy as np
import pandas as pd
import torch
#import PPMI_prediction_NN
import torch.nn as nn


###############################################################################

PROJECT = "PPMI"
INPUT_DATA_PATH = f"input_data/{PROJECT}/PPMI_cleaned_altered.csv"
MODEL_PATH= f"model/{PROJECT}/"
GLOBAL_MODEL_PATH = f"{MODEL_PATH}/GlobalModel.txt"
CLIENTS = 3

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
    
def get_one_vec_sorted_layers(model):
    layer_names = model.keys()
    size = 0
    for name in layer_names:
        size += model[name].view(-1).shape[0]
    sum_var = torch.FloatTensor(size).fill_(0)
    size = 0
    for name in layer_names:
        layer_as_vector = model[name].view(-1)
        layer_width = layer_as_vector.shape[0]
        sum_var[size:size + layer_width] = layer_as_vector
        size += layer_width
    return sum_var

def determine_number_of_entries_in_matrix(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result

def recover_model_from_vec(example_model, vec_to_recover, layer_names):
    result = {}
    start_index_of_next_layer = 0
    for layer_name in layer_names:
        expected_shape = example_model[layer_name].shape
        entries_in_layer = determine_number_of_entries_in_matrix(expected_shape)
        end_index_of_current_layer = start_index_of_next_layer + entries_in_layer
        entries = vec_to_recover[start_index_of_next_layer: end_index_of_current_layer]
        result[layer_name] = entries.view(expected_shape)
        start_index_of_next_layer += entries_in_layer
    return result





global_model = PPMIModel()
global_model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
global_model_vec = get_one_vec_sorted_layers(global_model.state_dict())
delta_wk_h_np = np.loadtxt(f'model/PPMI/Delta_0.txt')
vec = torch.tensor(delta_wk_h_np)

summed_deltas = vec[1:]
summed_h = vec[0]

for i in range(1, CLIENTS):
    #delta_file = f'Delta_{i:03d}.txt'
    delta_wk_h_np = np.loadtxt(f'model/PPMI/Delta_{i}.txt')
    vec = torch.tensor(delta_wk_h_np)
    summed_h += vec[0]
    summed_deltas += vec[1:]

h = torch.empty(len(summed_deltas)).fill_(summed_h)
#print(summed_deltas, "summed deltas")
#print(h, "h vec should all be same value")
print(torch.div(summed_deltas, h), "update")
result = torch.subtract(global_model_vec , (torch.div(summed_deltas, h)))
result_modell = recover_model_from_vec(global_model.state_dict(), result, global_model.state_dict().keys())
#print(result_modell)
torch.save(result_modell, GLOBAL_MODEL_PATH)