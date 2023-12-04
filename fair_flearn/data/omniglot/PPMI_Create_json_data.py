import scipy.io
import numpy as np
import random
import json
import pandas as pd
import torch.utils.data as data
import torch

NUM_USER = 5
torch.manual_seed(42)
generator = torch.Generator().manual_seed(42)



# preprocess data (x-mean)/stdev
def preprocess(x):
    means = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    
    x = (x - means) * 1.0 / std
    where_are_NaNs = isnan(x)
    x[where_are_NaNs] = 0
    return x

def my_random_split(dataset, num_splits):
    splits = []
    for i in range(num_splits):
        splits.append([])
    used_len = len(dataset) - len(dataset)%num_splits
    unused = []
    for i in range(used_len):
        unused.append(i)

    while(len(unused)>0):
        for i in range(num_splits):
            random_index = random.randint(0, len(unused)-1)
            item = unused[random_index]
            splits[i].append(dataset[item])
            unused.remove(item)
    print(splits)
    return splits

def create_users(fullset):
    num_samples_client = 1/NUM_USER
    
    splits = [num_samples_client] * NUM_USER 

    print(splits)
    #array_of_users= data.random_split(fullset, splits, generator=generator)
    array_of_users = my_random_split(fullset, NUM_USER)
    print("how many users?", len(array_of_users))
    for i in range(len(array_of_users)): 
        print(array_of_users[i])
    return array_of_users
    
    



def generate_data():
    X = []
    y = []
    mat = pd.read_csv('./raw_data/test.csv')
    mat = torch.Tensor(mat.to_numpy())
    mat = create_users(mat)
    raw_y = []
    raw_x = []
    for i in range(NUM_USER):
        print(mat[i])
        raw_x.append(mat[i][:, 2:])
        raw_y.append(mat[i][:, 1])
    print("raw_data x_0", raw_x[0])
    print("rows?", len(raw_x[0]))
    print("columns?", len(raw_x[0][0]))

    print("number of users:", len(raw_x), len(raw_y))
    print("number of features:", len(raw_x[0][0]))

    
    for i in range(NUM_USER):
        print("{}-th user has {} samples".format(i, len(raw_x[i][0][0])))
        #print(len(raw_x[i][0]) * 0.75)
        X.append(preprocess(raw_x[i][0]).tolist())
        y.append(raw_y[i].tolist())
        num = 0
        for j in range(len(raw_y[i][0])):
            if raw_y[i][0][j] == 1:
                num += 1
        print("ratio, ", num * 1.0 / len(raw_y[i][0]))
    return X, y
    

def main():


    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    train_path = "./data/train/mytrain.json"
    test_path = "./data/test/mytest.json"


    X, y = generate_data()

    
    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    for i in range(NUM_USER):

        uname = 'f_{0:05d}'.format(i)        
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.75 * num_samples) # the percentage of training samples is 0.75 (in order to be consistant with the statistics shown in the MTL paper)
        test_len = num_samples - train_len
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)
    

    with open(train_path,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)
    

if __name__ == "__main__":
    main()


