import numpy as np
import pandas as pd
import torch
#import PPMI_prediction_NN
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris


###############################################################################

PROJECT = "IRIS"
#INPUT_DATA_PATH = f"input_data/{PROJECT}/PPMI_cleaned_altered.csv"
MODEL_PATH= f"model/{PROJECT}/"
GLOBAL_MODEL_PATH = f"{MODEL_PATH}/GlobalModel.txt"
CLIENTS = 3

###############################################################################
features, labels = load_iris(return_X_y= True)


class IrisModel(nn.Module):
    def __init__(self, input_dim):
        super(IrisModel, self).__init__()
        self.layer1 = nn.Linear(input_dim,50)
        self.layer2 = nn.Linear(50, 20)
        self.layer3 = nn.Linear(20, 3)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x)) # To check with the loss function
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





global_model = IrisModel(features.shape[1])
global_model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
global_model_vec = get_one_vec_sorted_layers(global_model.state_dict())
delta_wk_h_np = np.loadtxt(f'model/IRIS/Delta_0.txt')
vec = torch.tensor(delta_wk_h_np)

summed_deltas = vec[1:]
summed_h = vec[0]

for i in range(1, CLIENTS):
    #delta_file = f'Delta_{i:03d}.txt'
    delta_wk_h_np = np.loadtxt(f'model/IRIS/Delta_{i}.txt')
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