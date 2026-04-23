import numpy as np

def tanh(x):

    x = np.array(x)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))    