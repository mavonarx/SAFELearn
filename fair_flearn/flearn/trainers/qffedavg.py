import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import sys as sys

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch


# Change constants here
###############################################################################
#MODE = int(sys.argv[1])  # 0 is training mode, 1 is eval mode, 2 is print params mode
#LIPSCHITZCONSTANT = 1000  # this should be: 1 / learning_rate (safelearn cant handle numbers this largen so we use 1) deprecated
#Q_FACTOR = 1 #deprecated
#TORCHSEED = int(sys.argv[2])
DEFAULT_DEVICE = "cpu"
NUMBER_OF_CLIENTS = 3
PROJECT = "PPMI"
INPUT_DATA_PATH = f"input_data/{PROJECT}/PPMI_cleaned_altered.csv"
MODEL_PATH= f"model/{PROJECT}/"
GLOBAL_MODEL_PATH = f"{MODEL_PATH}/GlobalModel.txt"
N_EPOCHS = 50
BATCH_SIZE = 64
###############################################################################


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset, round):
        print('Using fair fed avg to Train')
        self.inner_opt = tf.compat.v1.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.round = round

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))

        num_clients = len(self.clients)
        pk = np.ones(num_clients) * 1.0 / num_clients
        comunication_index = self.round

        #for i in range(self.num_rounds+1):
        #if i % self.eval_every == 0:
        num_test, num_correct_test = self.test() # have set the latest model for all clients
        num_train, num_correct_train = self.train_error()  
        num_val, num_correct_val = self.validate()  
        tqdm.write('At round {} testing accuracy: {}'.format(comunication_index, np.sum(np.array(num_correct_test)) * 1.0 / np.sum(np.array(num_test))))
        tqdm.write('At round {} training accuracy: {}'.format(comunication_index, np.sum(np.array(num_correct_train)) * 1.0 / np.sum(np.array(num_train))))
        tqdm.write('At round {} validating accuracy: {}'.format(comunication_index, np.sum(np.array(num_correct_val)) * 1.0 / np.sum(np.array(num_val))))
        
        
        #if i % self.log_interval == 0 and i > int(self.num_rounds/2):      TODO make things append to file instead of writing new file
        test_accuracies = np.divide(np.asarray(num_correct_test), np.asarray(num_test))
        np.savetxt(self.output + "_" + str(comunication_index) + "_test.csv", test_accuracies, delimiter=",")
        train_accuracies = np.divide(np.asarray(num_correct_train), np.asarray(num_train))
        np.savetxt(self.output + "_" + str(comunication_index) + "_train.csv", train_accuracies, delimiter=",")
        validation_accuracies = np.divide(np.asarray(num_correct_val), np.asarray(num_val))
        np.savetxt(self.output + "_" + str(comunication_index) + "_validation.csv", validation_accuracies, delimiter=",")
        
        
        indices, selected_clients = self.select_clients(round=comunication_index, pk=pk, num_clients=self.clients_per_round)

        selected_clients = selected_clients.tolist()
        
        for client_index, c in enumerate(selected_clients):
            Deltas = []
            hs = []
            # communicate the latest model
            c.set_params(self.latest_model)
            weights_before = c.get_params()
            loss = c.get_loss() # compute loss on the whole training data, with respect to the starting point (the global model)
            soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
            new_weights = soln[1]
            # plug in the weight updates into the gradient
            grads = [(u - v) * 1.0 / self.learning_rate for u, v in zip(weights_before, new_weights)]
            
            Deltas.append([np.float_power(loss+1e-10, self.q) * grad for grad in grads])
            
            # estimation of the local Lipchitz constant
            hs.append(self.q * np.float_power(loss+1e-10, (self.q-1)) * norm_grad(grads) + (1.0/self.learning_rate) * np.float_power(loss+1e-10, self.q))
        
            combined = np.concatenate((np.array([hs]), Deltas))
            np.savetxt(f"{MODEL_PATH}Delta_{client_index}.txt", combined, fmt='%.8f')
            
            print(weights_before)
        np.savetxt(f"{GLOBAL_MODEL_PATH}.txt", weights_before, fmt='%.8f')
        
        
        
        # aggregate using the dynamic step-size
        self.latest_model = self.aggregate2(weights_before, Deltas, hs)

                    



