import scipy.io
import numpy as np
import random
import json
from numpy import *
import pandas as pd
import torch.utils.data as data
import torch
from json import JSONEncoder
import os

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

def create_users(fullset:pd.DataFrame):
    
    num_samples_client = 1.0/NUM_USER
    
    splits = []
    for i in range(NUM_USER):
        splits.append(num_samples_client)
    a,s,d,f,g = data.random_split(fullset, splits, generator=generator)
    x = [a,s,d,f,g]
    return x
    
    
    

def generate_data():
    X = []
    y = []
    mat = pd.read_csv('./raw_data/PPMI_cleaned_altered.csv')
    #mat = torch.Tensor(mat.to_numpy())
    mat = mat.to_numpy()
    mat = create_users(mat)
    raw_y = [[],[],[],[],[]]
    raw_x = [[],[],[],[],[]]
    
    for i in range(NUM_USER):
        for index in mat[i].indices:
            raw_x[i].append(mat[i].dataset[index][2:])
            raw_y[i].append(mat[i].dataset[index][1])

    print("number of users:", len(raw_x), len(raw_y))
    print("number of features:", len(raw_x[0][0]))

    
    for i in range(NUM_USER):
        print("{}-th user has {} samples".format(i, len(raw_x[i])))
        #print(len(raw_x[i][0]) * 0.75)
        X.append((raw_x[i]))
        y.append(raw_y[i])
        num = 0
        for j in range(len(raw_y[i])):
            if raw_y[i][j] == 1:
                num += 1
        print("ratio, ", num * 1.0 / len(raw_y[i]))
    return X, y
    

def main():

    os.makedirs("data/train",exist_ok=True)
    os.makedirs("data/test",exist_ok=True)

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
        
        
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open(train_path,'w') as outfile:
        json.dump(train_data, outfile, cls=NumpyEncoder)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile, cls=NumpyEncoder)
    

if __name__ == "__main__":
    main()


