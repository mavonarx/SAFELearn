import numpy as np
import os, sys
import tensorflow as tf
from tqdm import trange

from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size, process_grad


class Model(object):
    '''
    Assumes that images are 28px by 28px
    '''

    def __init__(self, num_classes, q, optimizer, seed=1):

        # params
        self.num_classes = num_classes

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.compat.v1.set_random_seed(1 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(
                q, optimizer)
            self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())
            metadata = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.compat.v1.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, q, optimizer): # tried to rebuild ou own PPMI NN in TF
        """Model function for NN."""
        features = tf.compat.v1.placeholder(tf.float32, shape=[None, 12], name='features')
        labels = tf.compat.v1.placeholder(tf.int64, shape=[None], name='labels')
        input_layer = tf.reshape(features, (-1,12))
        
        dense1 = tf.compat.v1.layers.dense(inputs=input_layer, units=20,activation=tf.nn.relu)
        dense2 = tf.compat.v1.layers.dense(inputs=dense1, units=15,activation=tf.nn.relu)
        dense3 = tf.compat.v1.layers.dense(inputs=dense2, units=12,activation=tf.nn.relu)
        dense4 = tf.compat.v1.layers.dense(inputs=dense3, units=4,activation=tf.nn.relu)
        logits = tf.compat.v1.layers.dense(inputs=dense4, units=self.num_classes)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.compat.v1.train.get_global_step())

        eval_metric_ops = tf.math.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.compat.v1.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.compat.v1.trainable_variables())
        return model_params

    def get_gradients(self, data, latest_model):

        '''
        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                feed_dict={self.features: data['x'], self.labels: data['y']})
        return model_grads
        '''
        model_len = process_grad(latest_model).size
        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                                        feed_dict={self.features: data['x'], self.labels: data['y']})
            grads = process_grad(model_grads)

        return num_samples, grads

    def get_loss(self, data):
        with self.graph.as_default():
            loss = self.sess.run(self.loss, feed_dict={self.features: data['x'], self.labels: data['y']})
        return loss

    def solve_sgd(self, mini_batch_data):
        with self.graph.as_default():
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
                                           feed_dict={self.features: mini_batch_data[0],
                                                      self.labels: mini_batch_data[1]})

        weights = self.get_params()
        return grads, loss, weights

    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem'''
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                                  feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return soln, comp

    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss

    def close(self):
        self.sess.close()
