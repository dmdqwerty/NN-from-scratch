import numpy as np

def weighted_sum(x, w):
    x = np.array(x)
    w = np.array(w)
    return np.dot(x, w)


if __name__ == "__main__":
    print(weighted_sum([1], [4]))