import torch
import os
import numpy as np
import sys

torch.manual_seed(42)

PUSH_FACTOR = 2 ** 10
LIMIT = (2 ** 3) * PUSH_FACTOR

if(len(sys.argv)==1):
    print("For splitting use: python3", sys.argv[0], "split\nFor aggregating use: python3", sys.argv[0], "combine")


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


def restrict_values(vec):
    vec = PUSH_FACTOR * vec
    vec = torch.round(vec)
    vec = torch.clamp(vec, -LIMIT, LIMIT)
    restricted_vec = vec.type(torch.LongTensor)
    return restricted_vec


def unrestrict_values(recovered_restricted_vec):
    recovered_restricted_vec = recovered_restricted_vec.type(torch.FloatTensor)
    return recovered_restricted_vec / PUSH_FACTOR
    

def split(restricted_vec):
    a = torch.LongTensor(restricted_vec.shape).random_(-LIMIT, LIMIT)
    b = restricted_vec - a
    safety_counter = 0
    while True:
        indices_to_recompute = torch.nonzero(torch.abs(b) >= LIMIT)
        if len(indices_to_recompute) == 0:
            break
        if safety_counter > 100:
            raise Exception('Did not find suitable randomvalues')
        indices_to_recompute = indices_to_recompute.view(-1)
        print(f'\tRegenerate {indices_to_recompute.shape[0]} elements (from {restricted_vec.shape[0]})')
        a[indices_to_recompute] = torch.LongTensor(restricted_vec[indices_to_recompute].shape).random_(-LIMIT, LIMIT)
        b = restricted_vec - a
        safety_counter += 1
    return a, b


def create_splits(directory_name, global_model_path, local_model_paths):
    splitted_file_dir = "data/"+directory_name+"Splits"
    if not os.path.exists(splitted_file_dir):
        os.mkdir(splitted_file_dir)
    global_model = torch.load(global_model_path)
    global_model_as_vec = get_one_vec_sorted_layers(global_model)
    restricted_vec = restrict_values(global_model_as_vec)    
    np.savetxt(splitted_file_dir + '/global.txt', restricted_vec.numpy(), fmt='%d')
    
    for i, model_path in enumerate(local_model_paths):
        local_model = torch.load(model_path)
        local_model_as_vec = get_one_vec_sorted_layers(local_model)
        restricted_local_vec = restrict_values(local_model_as_vec)    
        a, b = split(restricted_local_vec)
        a_file = f'{splitted_file_dir}/A_C{i:03d}.txt'
        b_file = f'{splitted_file_dir}/B_C{i:03d}.txt'
        np.savetxt(a_file, a.numpy(), fmt='%d')
        np.savetxt(b_file, b.numpy(), fmt='%d')
    print(a_file)
    print(b_file)


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


def determine_aggregated_model(old_global_model_path, path_to_share1, path_to_share2):
    old_global_model = torch.load(old_global_model_path)
    layer_names = old_global_model.keys()
    share1 = np.loadtxt(path_to_share1, dtype=np.int64)
    share2 = np.loadtxt(path_to_share2, dtype=np.int64)
    restricted_vec = share1 + share2

    unrestricted_vec = unrestrict_values(torch.from_numpy(restricted_vec))
    return recover_model_from_vec(old_global_model, unrestricted_vec, layer_names)

if (len(sys.argv) >1 and sys.argv[1] == "split"):
    print("Splitting to data/MyTestDir")
    localmodelpaths = []
    localmodelpaths.append("./model/LocalModel")
    create_splits("MyTestDir","./model/GlobalModel",localmodelpaths)

newModelPath = "./model/NewModel"
if (len(sys.argv) >1 and sys.argv[1] == "combine"):
    print("Aggregating! - new model will be saved at", newModelPath)
    newmodel = determine_aggregated_model("./model/GlobalModel", "./data/Aggregated/AggregatedModel_A.txt", "./data/Aggregated/AggregatedModel_B.txt")
    torch.save(newmodel, newModelPath)