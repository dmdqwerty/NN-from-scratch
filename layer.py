import numpy as np
from neuron import Neuron

class Layer:
    def __init__(self, num_neurons, input_size, activation_func, entry_func, learning_rate):
        self.num_neurons = num_neurons
        self.neurons = []

        # Створення нейронів у шарі
        for _ in range(num_neurons):
            neuron = Neuron(
                entry_func=entry_func,
                activation_func=activation_func,
                input_len=input_size,
                learning_rate=learning_rate
            )
            self.neurons.append(neuron)

    def forward(self, input_data):
        """Пряме поширення: обчислює вихід для кожного нейрона у шарі."""
        input_vector = np.array(input_data, dtype=float).flatten()
        output = []
        for neuron in self.neurons:
            output_val = neuron.forward(input_vector).item()
            output.append(output_val)

        return np.array(output)

    def backward(self, errors):
        """Зворотне поширення: обчислює і поширює помилку на попередній шар."""
        error_to_propagate = np.zeros(self.neurons[0].input_len)

        for i, neuron in enumerate(self.neurons):
            neuron_error = errors[i]
            delta_w = neuron.backward(neuron_error)
            error_to_propagate += delta_w

        return error_to_propagate