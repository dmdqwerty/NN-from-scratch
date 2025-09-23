import numpy as np

def ReLU(z):
    z = np.array(z)
    return max(0, z)


def sigmoid(z, a=1):
    z = np.array(z)
    return 1 / (1 + np.exp(-a * z))

