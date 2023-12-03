import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sys
import sklearn.metrics as sklm
import os
import glob
from torcheval.metrics.functional import multiclass_f1_score, multiclass_auroc
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import torch.nn.functional as F

import sys
np.set_printoptions(threshold=sys.maxsize)

# Change constants here
###############################################################################
MODE = int(sys.argv[1])  # 0 is training mode, 1 is eval mode, 2 is print params mode
LIPSCHITZCONSTANT = 1000  # this should be: 1 / learning_rate (safelearn cant handle numbers this largen so we use 1)
Q_FACTOR = 1
TORCHSEED = int(sys.argv[2])
DEFAULT_DEVICE = "cpu"
NUMBER_OF_CLIENTS = 3
PROJECT = "IRIS"
#INPUT_DATA_PATH = f"input_data/{PROJECT}/PPMI_cleaned_altered.csv"
MODEL_PATH= f"model/{PROJECT}/"
GLOBAL_MODEL_PATH = f"{MODEL_PATH}/GlobalModel.txt"
N_EPOCHS = 10
BATCH_SIZE = 64
###############################################################################

# INIT
###############################################################################
torch.manual_seed(TORCHSEED)
generator = torch.Generator().manual_seed(TORCHSEED)

eval_generator = torch.Generator().manual_seed(42)

device = torch.device(DEFAULT_DEVICE)
#print("Device:", device)

if not os.path.exists("model"):
        os.mkdir("model")

if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
###############################################################################
if (MODE == 2):
    print("Q_FACTOR ",Q_FACTOR , ", TORCHSEED ",  TORCHSEED , ", Nr. of Clients ", NUMBER_OF_CLIENTS, ", N_EPOCHS ", N_EPOCHS, ", Batch Size ", BATCH_SIZE, "LIPSCHITZ:", LIPSCHITZCONSTANT)
    exit()

# initalized an np array for storing the loss of each clients data in respect to the current global model without training of the client itself. 
GLOBAL_LOSS = np.zeros(NUMBER_OF_CLIENTS) 

features, labels = load_iris(return_X_y= True)




clients_features = []
clients_labels = []


features_train,features_eval, labels_train, labels_eval = train_test_split(features, labels, random_state=42, shuffle=True)
x_eval, y_eval = Variable(torch.from_numpy(features_eval)).float(), Variable(torch.from_numpy(labels_eval)).long()

# Split the data into non-overlapping parts
split_size = len(features_train) // NUMBER_OF_CLIENTS 
for client_index in range(NUMBER_OF_CLIENTS):
    split_features = features_train[client_index * split_size: (client_index + 1) * split_size]
    split_labels = labels_train[client_index * split_size: (client_index + 1) * split_size]
    clients_features.append(split_features)
    clients_labels.append(split_labels)

client_loss = np.zeros(NUMBER_OF_CLIENTS) 



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
    
    
    
    
def eval_model(model, X_test, y_test):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()


    import numpy as np
    np.set_printoptions(threshold=np.inf)


    with torch.no_grad():
        y_pred = model(X_test)
        
        print(y_eval, ", y_eval")
        print(torch.argmax(torch.nn.functional.softmax(y_pred, dim=1), dim=1).numpy(), ", y_pred")
        print(len(features), ", length of iris")
        #print(y_test.flatten().numpy())
        f1 = multiclass_f1_score(y_pred, torch.reshape( y_test, (-1, )), num_classes=3).numpy()
        auroc = multiclass_auroc(y_pred, torch.reshape( y_test, (-1, )), num_classes=3).numpy()
        print ("The accuracy is", accuracy_score(y_test, np.argmax(y_pred, axis=1)))
        print("F1 Score:", f1)
        print("AUROC:", auroc)
        


if (MODE == 1):
    model = IrisModel(features_train.shape[1])
    # if there exists a global model from earlier learnings import it
    model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
    
    model.eval()

    eval_model(model, x_eval, y_eval)

     
    #y_pred_eval = model(X_eval)
    
    #print(torch.argmax(torch.nn.functional.softmax(y_pred_eval, dim=1), dim=1).numpy())
    #print(y_eval.flatten().numpy())
    
    #print(multiclass_f1_score(y_pred_eval, y_eval, num_classes=3).numpy(), ",")
    #print(multiclass_auroc(y_pred_eval, y_eval, num_classes=3).numpy(), ",")
    
    #eval_model(model, X_eval, y_eval, 1)
    exit()




# if the global model does not yet exist create a new fully untrained one
if not os.path.exists(GLOBAL_MODEL_PATH):
    model = IrisModel(features_train.shape[1])
    torch.save(model.state_dict(), GLOBAL_MODEL_PATH)

global_model = IrisModel(features_train.shape[1])
global_model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))

def delete_files(file_pattern):
    files_to_delete = glob.glob(file_pattern)
    for file in files_to_delete:
        os.remove(file)

# delete all existing client models, losses, and delta files
delete_files(f"{MODEL_PATH}Model_*")
delete_files(f"{MODEL_PATH}Loss_*")
delete_files(f"{MODEL_PATH}Delta_*")




for client_index, clients_features in enumerate(clients_features):
    clients_targets = clients_labels[client_index]
    train_size = int(0.8 * len(clients_features))
    test_size = len(clients_features) - train_size
    
    train_features, test_features, train_labels, test_labels = train_test_split(features_train, labels_train, random_state=42, shuffle=True)
    
    
    x_train, y_train = Variable(torch.from_numpy(train_features)).float(), Variable(torch.from_numpy(train_labels)).long()
    x_test, y_test = Variable(torch.from_numpy(test_features)).float(), Variable(torch.from_numpy(test_labels)).long()
        
    model = IrisModel(features_train.shape[1])
        
    # if there exists a global model from earlier learnings import it
    model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
    loss_fn = nn.CrossEntropyLoss()
    

    y_pred_loss = model(x_train)
    GLOBAL_LOSS[client_index] = loss_fn(y_pred_loss, y_train).detach().numpy()

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    def train_model(model, optimizer, X_train, y_train, loss_fn, n_epochs=100):

        model.train()
        model.to(device)
                    
        for epoch in range(n_epochs):
            for i in range(0, len(X_train), BATCH_SIZE):
                Xbatch = X_train[i:i+BATCH_SIZE].to(device)
                ybatch = y_train[i:i+BATCH_SIZE].to(device)
                y_pred = model(Xbatch)
                loss = loss_fn(y_pred, ybatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #print(f'Client {client_index}, Epoch {epoch}, latest loss {loss}')
        torch.save(model.state_dict(), f"{MODEL_PATH}Model_{client_index}.txt")
        #torch.save(loss_fn(y_pred, torch.reshape(y_train, (-1,)).to(torch.int64)), f"model/PPMImodels/Loss_{client_index}"
    ## execute the code

    if (MODE == 0):
        train_model(model, optimizer, x_train, y_train, loss_fn, N_EPOCHS)
    eval_model(model, x_test, y_test)


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



def norm_grad(grad_list):
    # input: nested gradients
    # output: square of the L-2 norm

    client_grads = grad_list[0] # shape now: (784, 26)

    for i in range(1, len(grad_list)):
        client_grads = np.append(client_grads, grad_list[i]) # output a flattened array

    return np.sum(np.square(client_grads))



def calculate_delta_wt(global_model, model, L):
    vec_glob = get_one_vec_sorted_layers(global_model.state_dict())
    vec_mod = get_one_vec_sorted_layers(model.state_dict())
    return torch.mul((torch.subtract(vec_glob,  vec_mod)), L)

def calculate_delta(q, loss, deltawt):
    return torch.mul(deltawt, loss**q)

def calculate_ht(q, loss, deltawt, L):
    return q * (loss ** (q-1)) * np.linalg.norm(deltawt.detach().numpy(),2) ** 2 + (loss ** q) * L

if (MODE == 0):

    for client_index in range((NUMBER_OF_CLIENTS)):
        model = IrisModel(features_train.shape[1])
        #print(client_loss[client_index])
        model.load_state_dict(torch.load(f"{MODEL_PATH}Model_{client_index}.txt"))
        deltawt = calculate_delta_wt(global_model, model, LIPSCHITZCONSTANT)
        #print(deltawt, "deltawt")
        delta = calculate_delta(Q_FACTOR, GLOBAL_LOSS[client_index], deltawt)
        ht = calculate_ht(Q_FACTOR, GLOBAL_LOSS[client_index], deltawt, LIPSCHITZCONSTANT)
        #print(ht, "printing ht")
        combined = np.concatenate((np.array([ht]), delta.detach().numpy()))
        np.savetxt(f"{MODEL_PATH}Delta_{client_index}.txt", combined, fmt='%.8f')
        #f.write(delta.numpy() + "\n" + ht.numpy())
        #f.close()
    print(GLOBAL_LOSS) 
        
