import numpy as np

def linear_layer_forward(X, W, b):

    X = np.asarray(X)
    W = np.asarray(W)
    b = np.asarray(b)

    Y = (X @ W) + b

    return Y.tolist()