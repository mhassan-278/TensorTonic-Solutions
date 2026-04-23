import numpy as np

def swish(x):

    x = np.array(x)
    sig_x = 1 / (1 + np.exp(-x))

    return x * sig_x