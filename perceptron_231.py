import numpy as np
from layer import Layer
from activation_functions import identity
from entry_functions import weighted_sum


class Perceptron231:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.epochs = 500

        self.hidden_layer = Layer(
            num_neurons=3,
            input_size=2,
            activation_func=identity,
            entry_func=weighted_sum,
            learning_rate=self.learning_rate
        )

        self.output_layer = Layer(
            num_neurons=1,
            input_size=3,
            activation_func=identity,
            entry_func=weighted_sum,
            learning_rate=self.learning_rate
        )

    def forward(self, x):
        x_vector = np.array(x, dtype=float).flatten()

        hidden_output = self.hidden_layer.forward(x_vector)

        final_output = self.output_layer.forward(hidden_output)

        return final_output.item()

    def train_on_one_example(self, x_input, y_true):
        x_vector = np.array(x_input, dtype=float).flatten()

        self.hidden_layer.forward(x_vector)

        l1_output = np.array([n.last_output.item() for n in self.hidden_layer.neurons])
        y_pred = self.output_layer.forward(l1_output)

        error = y_pred - y_true
        loss = error ** 2

        error_to_L1 = self.output_layer.backward(errors=error)

        self.hidden_layer.backward(errors=error_to_L1)

        return y_pred, loss


if __name__ == "__main__":

    X_data = np.array([
        [1.0, 2.0],
        [0.5, 0.5],
        [3.0, 1.0],
        [0.2, 0.7],
    ])
    Y_data = np.array([3.0, 1.0, 4.0, 0.9])

    X_test = [1.5, 2.5]
    Y_EXPECTED = 4.0

    model = Perceptron231(learning_rate=0.01)
    num_samples = len(X_data)

    print("Training Progress (ON-LINE SGD):")
    print(f"{'Epoch':<6} {'Last_Y_Pred':>25} {'Avg_Loss (MSE)':>35}")

    for epoch in range(model.epochs):
        indices = np.random.permutation(num_samples)
        total_loss = 0

        for i in indices:
            x_sample = X_data[i]
            y_target = Y_data[i]

            y_pred, loss = model.train_on_one_example(x_sample, y_target)
            total_loss += loss.item()

        avg_loss = total_loss / num_samples

        if epoch % (model.epochs // 10) == 0:
            last_y_pred_scalar = y_pred.item()
            avg_loss_scalar = avg_loss

            print(f"{epoch:<6} {last_y_pred_scalar:>25f} {avg_loss_scalar:>35e}")

    print("\nFinal Parameters and Recognition:")

    print("\n  Layer 1 (Hidden, 3 Neurons):")
    for i, neuron in enumerate(model.hidden_layer.neurons):
        print(f"    Neuron {i}: W={neuron.weights}, B={neuron.bias.item()}")

    print("\n  Layer 2 (Output, 1 Neuron):")
    print(f"    Final Weights: {model.output_layer.neurons[0].weights}")
    print(f"    Final Bias:    {model.output_layer.neurons[0].bias.item()}")

    test_pred = model.forward(X_test)

    print("\n  Test Input:")
    print(f"    X: {X_test}")
    print(f"    Expected Y: {Y_EXPECTED}")
    print(f"    Predicted Y: {test_pred}")