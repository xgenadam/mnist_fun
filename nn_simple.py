"""
This is a workthrough of the basic back propagation alogorithm as described here: http://neuralnetworksanddeeplearning.com/
The author here would like to thank the author of the aformentioned website as it was found to be helpful in developing understanding.
"""

import numpy as np
from itertools import chain
import pickle
from time import time

from utils import batch
from utils import MNISTDataSingleton

from sklearn.metrics import f1_score

@np.vectorize
def sigmoid_function(x):
    return float(1/(1 + np.exp(-x)))


def sigmoid_prime(output_array):
    assert len(output_array.shape) == 1
    one_array = np.array([1.0 for x in output_array])
    return output_array * (one_array - output_array)

@np.vectorize
def initial_cost_differential(pred, target):
    return pred - target


class FFNN(object):
    """
    feed forward neural network object
    """
    def __init__(self, layer_dimensions, learning_rate=0.05, random_weight_seed=42):
        np.random.seed(random_weight_seed)
        self.num_layers = len(layer_dimensions)
        self.weights = [np.random.rand(i, j) for i, j in zip(layer_dimensions[:-1], layer_dimensions[1:])]
        self.biases = [np.random.rand(layer_size) for layer_size in layer_dimensions]
        self.activation_function = sigmoid_function
        self.activation_prime = sigmoid_prime
        self.learning_rate = learning_rate

    def fit(self, X_train, y_train, batch_size):
        for b in batch(zip(X_train, y_train), batch_size, len(X_train)):
            nabla_bs = [np.zeros(b.shape) for b in self.biases]
            nabla_ws = [np.zeros(w.shape) for w in self.weights]

            for b_x, b_y in b:
                target_array = np.array([0.0 for x in range(10)], dtype=float)
                target_array[int(b_y)] = 1.0
                n_bs_latest, n_ws_latest = self.get_nablas(b_x, target_array)


                for layer_index, (layer_nab_b, new_layer_nab_bs) in enumerate(zip(nabla_bs, n_bs_latest)):
                    layer_nab_b += new_layer_nab_bs
                    nabla_bs[layer_index] = layer_nab_b

                for layer_index, (layer_nab_w, new_layer_nab_ws) in enumerate(zip(nabla_ws, n_ws_latest)):
                    layer_nab_w += new_layer_nab_ws
                    nabla_ws[layer_index] = layer_nab_w

            for layer_index, (weight_layer, nabla_w_layer) in enumerate(zip(self.weights, nabla_ws)):
                self.weights[layer_index] = weight_layer -   ((nabla_w_layer/float(len(b))) + self.learning_rate)

            for layer_index, (bias_layer, nabla_b_layer) in enumerate(zip(self.biases, nabla_bs)):
                self.biases[layer_index] = bias_layer - ((nabla_b_layer/float(len(b))) * self.learning_rate)

    def get_nablas(self, x, y):
        activation = self.activation_function(x - self.biases[0])
        layer_outputs = [activation, ]
        layer_output_primes = [self.activation_prime(activation)]
        for index, (weights, biases) in enumerate(zip(self.weights, self.biases[1:])):
            activation = self.predict_layer(activation, weights, biases)
            layer_outputs.append(activation)
            layer_output_prime = self.activation_prime(activation)
            layer_output_primes.append(layer_output_prime)
            assert type(layer_output_prime) is np.ndarray, "this should be a 1D array!, type is {}".format(type(layer_output_prime))
            assert layer_output_prime.shape == biases.shape, "these should have the same shape!"

        # calculate the partial derivatives per layer with respect to the layer output
        dc_dhs_reverse = [initial_cost_differential(activation, y)]
        for layer_output_prime, weights in zip(layer_output_primes[-1::-1], chain(self.weights[-1::-1], [None])):
            if weights is not None:
                prev_dc_dh = dc_dhs_reverse[-1]
                dc_dhs_reverse.append(np.matmul(weights, prev_dc_dh * layer_output_prime))
            else:
                pass

        # now that we have the partial derivatives per layer with respect to node output per layer
        # we can start on building the nablas
        nabla_bs = []
        nabla_ws = []
        assert len(layer_outputs) == len(layer_output_primes) == len(dc_dhs_reverse) == 3
        for layer_index, (dc_dh, layer_input, layer_output_prime) in enumerate(zip(dc_dhs_reverse[-1::-1], chain([None, ], layer_outputs[:-1]), layer_output_primes)):
            assert dc_dh.shape == layer_output_prime.shape, "shape mismatch on nabla biases {}".format(layer_index)
            nabla_b = dc_dh * layer_output_prime
            nabla_bs.append(nabla_b)
            if layer_index > 0:
                nabla_w = []
                for input in layer_input:
                    # assert input.shape == nab_b_i.shape, "shape mismatch on nabla weights {}".format(layer_index)
                    w_ij = input * nabla_b
                    # import pdb; pdb.set_trace()
                    nabla_w.append(w_ij)

                nabla_w = np.array(nabla_w, dtype=float)
                assert nabla_w.shape == self.weights[layer_index - 1].shape, \
                    "shape mismatch on nabla weights {}, shape is{}, expected shape is {}"\
                        .format(layer_index,
                                nabla_w.shape,
                                self.weights[layer_index -1].shape)

                nabla_ws.append(nabla_w)

        return nabla_bs, nabla_ws

    def predict(self, X):
        layer_input = self.activation_function(np.array(X, dtype=float) + self.biases[0])
        for weights, biases in zip(self.weights, self.biases[1:]):
            layer_input = self.predict_layer(layer_input, weights, biases)

        return layer_input

    def predict_layer(self, activation, weights, biases):
        return self.activation_function(np.matmul(activation, weights) + biases)


def nn_simple():
    X_train = MNISTDataSingleton().X_train
    y_train = MNISTDataSingleton().y_train

    X_test = MNISTDataSingleton().X_test
    y_test = MNISTDataSingleton().y_test

    batch_size = 50
    layer_dimensions = [len(X_train[0]), 15, 10]

    nn = FFNN(layer_dimensions)
    print('training classifier')
    t0 = time()
    nn.fit(X_train, y_train, batch_size)
    print('done_fitting')
    print('classifier trained in time {}'.format(time() - t0))

    y_pred = [float(np.argmax(nn.predict(X_input))) for X_input in X_test]

    score = f1_score(y_pred=y_pred, y_true=y_test, labels=list(range(10)), average=None)

    print('score is {}'.format(score))

    print('saving model')
    with open('./simple_neural_network_clf.pkl', 'w+') as f:
        pickle.dump(nn, f)
    print('model saved')

if __name__ == '__main__':
    nn_simple()