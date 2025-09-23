import numpy as np
from entry_functions import *
from activation_functions import *

class Neuron():
    def __init__(self, entry_func, activation_func, input_len=None, weights=None, bias=None):
        if weights is None:
            weights = np.random.uniform(low=-1.0, high=1.0, size=input_len if input_len else 1)
        
        if bias is None:
            bias = np.random.uniform(low=-1.0, high=1.0, size=1)

        self.entry_func = entry_func
        self.activation_func=activation_func
        self.weights = np.array(weights)
        self.bias = np.array(bias)
        self.input_len = len(self.weights)

    def forward(self, x):
        return self.activation_func(self.entry_func(x, self.weights) + self.bias)
    

if __name__ == "__main__":
    entry_data = [1, 2, 3, 4, 5]

    my_neuron = Neuron(entry_func=weighted_sum, activation_func=sigmoid, input_len=5)
    print(my_neuron.bias)
    print(my_neuron.forward(entry_data))


