import numpy as np

def  relu(x):

    x = np.array(x)
    return np.maximum(0,x)    