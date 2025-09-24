import numpy as np
from layer import Layer
from activation_functions import sigmoid
from entry_functions import weighted_sum


class Perceptron111:

    def __init__(self, learning_rate=1.5):
        self.learning_rate = learning_rate
        self.epochs = 1000

        self.hidden_layer = Layer(
            num_neurons=1,
            input_size=1,
            activation_func=sigmoid,
            entry_func=weighted_sum,
            learning_rate=self.learning_rate
        )

        self.output_layer = Layer(
            num_neurons=1,
            input_size=1,
            activation_func=sigmoid,
            entry_func=weighted_sum,
            learning_rate=self.learning_rate
        )

    def forward(self, x):
        x_vector = np.array([x])
        hidden_output = self.hidden_layer.forward(x_vector)
        final_output = self.output_layer.forward(hidden_output)

        return final_output.item()

    def train_on_one_example(self, x, y_true):
        x_vector = np.array([x])
        self.hidden_layer.forward(x_vector)
        y_pred = self.output_layer.forward(self.hidden_layer.neurons[0].last_output)

        error = y_pred - y_true
        loss = error ** 2

        error_to_L1 = self.output_layer.backward(errors=error)
        self.hidden_layer.backward(errors=error_to_L1)

        return y_pred, loss


if __name__ == "__main__":

    X = 0.8
    Y_TRUE = 0.1
    X_test = 0.3
    model = Perceptron111(learning_rate=1.5)

    print("Training Progress (SGD):")
    print(f"{'Epoch':<6} {'Y_Pred':>25} {'Loss (MSE)':>35}")

    for epoch in range(model.epochs):
        y_pred, loss = model.train_on_one_example(X, Y_TRUE)

        if epoch % (model.epochs // 10) == 0:
            y_pred_scalar = y_pred.item()
            loss_scalar = loss.item()

            print(f"{epoch:<6} {y_pred_scalar:>25f} {loss_scalar:>35e}")

    print("\nFinal Parameters and Recognition:")

    print(f"  Final Weights (L1): {model.hidden_layer.neurons[0].weights}")
    print(f"  Final Bias (L1):    {model.hidden_layer.neurons[0].bias.item()}")

    print(f"  Final Weights (L2): {model.output_layer.neurons[0].weights}")
    print(f"  Final Bias (L2):    {model.output_layer.neurons[0].bias.item()}")

    test_pred = model.forward(X_test)

    print(f"  Test Input: {X_test}")
    print(f"  Test Prediction: {test_pred}")