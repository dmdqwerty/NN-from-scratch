import numpy as np
from entry_functions import weighted_sum
from activation_functions import identity

class Neuron():
    def __init__(self, *, 
                 entry_func, 
                 activation_func,
                 input_len=None, 
                 weights=None, 
                 bias=None, 
                 learning_rate=0.1):
        if weights is None:
            weights = np.random.uniform(low=-1.0, high=1.0, size=input_len if input_len else 1)
        
        if bias is None:
            bias = np.random.uniform(low=-1.0, high=1.0, size=1)

        self.entry_func = entry_func
        self.activation_func=activation_func
        self.activation_derivative = activation_func.derivative
        self.weights = np.array(weights)
        self.bias = np.array(bias)
        self.input_len = len(self.weights)
        self.learning_rate = learning_rate
        # Допоміжні поля для режиму навчання
        self.last_input = None
        self.last_z = None
        self.last_output = None


    def forward(self, x):
        self.last_input = np.array(x)
        self.last_z = self.entry_func(self.last_input, self.weights) + self.bias
        self.last_output = self.activation_func(self.last_z)
        return self.last_output
    

    def backward(self, error):
        d_activation = self.activation_derivative(self.last_z)
        delta = error * d_activation  

        d_weights = delta * self.last_input
        d_bias = delta

        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias

        return delta * self.weights
    


if __name__ == "__main__":
    X = [2, 3]
    y_true = 5

    neuron = Neuron(entry_func=weighted_sum, activation_func=identity, input_len=2, learning_rate=0.01)
    for epoch in range(100):
        y_pred = neuron.forward(X)
        error = y_pred - y_true
        neuron.backward(error)

        if epoch % 10 == 0:
            loss = error**2
            print(f"Epoch {epoch}: y_pred={y_pred}, loss={loss}")

print("Final weights:", neuron.weights)
print("Final bias:", neuron.bias)
print("Test prediction:", neuron.forward([4, 7]))

