"""
This is a workthrough of the basic back propagation alogorithm as described here: http://neuralnetworksanddeeplearning.com/
The author here would like to thank the author of the aformentioned website as it was found to be helpful in developing understanding.
"""

import numpy as np

from utils import batch
from utils import MNISTDataSingleton


@np.vectorize
def sigmoid_function(x):
    return float(1/(1 + np.exp(-x)))

@np.vectorize
def sigmoid_prime(x):
    sig = sigmoid_function(x)
    return float(sig * (1 - sig))

@np.vectorize
def initial_cost_differential(pred, target):
    return float(pred - target)


class FFNN(object):
    """
    feed forward neural network object
    """
    def __init__(self, layer_dimensions, learning_rate=0.05, activation_function=None, activation_prime=None, random_weight_seed=42):
        np.random.seed(random_weight_seed)
        self.num_layers = len(layer_dimensions)
        self.weights = [np.random.rand(i, j) for i, j in zip(layer_dimensions[:-1], layer_dimensions[1:])]
        self.biases = [np.random.rand(layer_size) for layer_size in layer_dimensions[1:]]
        self.activation_function = activation_function if activation_function is not None else sigmoid_function
        self.activation_prime = activation_prime if activation_function is not None else sigmoid_prime
        self.learning_rate = learning_rate

    def fit(self, X_train, y_train, batch_size):
        for b in batch(zip(X_train, y_train), batch_size, len(X_train)):
            nabla_bs = [np.zeros(b.shape) for b in self.biases]
            nabla_ws = [np.zeros(w.shape) for w in self.weights]

            for b_x, b_y in b:
                n = self.get_nablas(b_x, b_y)

    def get_nablas(self, x, y):
        nabla_bs = [np.zeros(b.shape) for b in self.biases]
        nabla_ws = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        activation_primes = []
        for index, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            activation = self.predict_layer(activation, weights, biases)
            activations.append(activation)
            activation_primes.append(self.activation_prime(activation))

        dc_dh_reversed = [initial_cost_differential(activation, y)]

        for activation_prime in activation_primes[-1::-1]:
            dc_dh_prev = dc_dh_reversed[-1]
            # TODO: this is wrong
            dc_dh_curr = None # np.multiply(dc_dh_prev, activation_prime)
            dc_dh_reversed.append(dc_dh_curr)

        for index, (nabla_b, nabla_w, activation, dc_dh) in enumerate(zip(nabla_bs, nabla_ws, activations, dc_dh_reversed[-1::-1])):
            import pdb; pdb.set_trace()
            nabla_b += None
            nabla_w += None

        return nabla_ws, nabla_bs

    def predict(self, X):
        layer_input = np.array(X, dtype=float)
        for weights, biases in zip(self.weights, self.biases):
            layer_input = self.predict_layer(layer_input, weights, biases)

        return layer_input

    def predict_layer(self, activation, weights, biases):
        return self.activation_function(np.matmul(activation, weights) - biases)


def nn_simple():
    X_train = MNISTDataSingleton().X_train
    y_train = MNISTDataSingleton().y_train

    X_test = MNISTDataSingleton().X_test
    y_test = MNISTDataSingleton().y_test

    batch_size = 50
    layer_dimensions = [len(X_train[0]), 15, 10]
    nn = FFNN(layer_dimensions)

    nn.fit(X_train, y_train, batch_size)


if __name__ == '__main__':
    nn_simple()