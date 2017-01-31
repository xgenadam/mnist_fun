"""
This is a workthrough of the basic back propagation alogorithm as described here: http://neuralnetworksanddeeplearning.com/
The author here would like to thank the author of the aformentioned website as it was found to be helpful in developing understanding.
"""

import numpy as np

from utils import batch
from utils import MNISTDataSingleton


def sigmoid_function(x):
    return float(1/(1 + np.exp(-x)))


def quardratic_cost(pred, target):
    return float(sum((target - pred)**2)/2)


class FFNN(object):
    """
    feed forward neural network object
    """
    def __init__(self, layer_dimensions, cost_function=None, activation_function=None, random_weight_seed=42):
        np.random.seed(random_weight_seed)
        self.weights = [np.random.rand(layer_size[i+1], layer_size) for i, layer_size in enumerate(layer_dimensions[:-1])]
        self.biases = [np.random.rand(layer_size) for layer_size in layer_dimensions]
        self.cost_function = cost_function if cost_function is not None else quardratic_cost
        self.activation_function = activation_function if activation_function is not None else sigmoid_function

    def fit(self, X_train, y_train, batch_size):
        for b in batch(zip(X_train, y_train), batch_size):
            pass

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
    layer_dimensions = [len(X_train[0]), 744, 10]
    nn = FFNN(layer_dimensions)

    nn.fit(X_train, y_train, batch_size)


if __name__ == '__main__':
    nn_simple()