import numpy as np

class Activation:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative

    def __call__(self, z):
        return self.func(z)


sigmoid = Activation(
    func=lambda z: 1 / (1 + np.exp(-z)),
    derivative=lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))
)


relu = Activation(
    func=lambda z: np.maximum(0, z),
    derivative=lambda z: (z > 0).astype(float)
)


identity = Activation(
    func=lambda z: z,
    derivative=lambda z: 1
)

