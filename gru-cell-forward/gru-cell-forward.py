import numpy as np

# Inputs
x1 = np.zeros((2,3))
h_prev1 = np.array([[1.0,-1.0],[2.0,0.0]])
x2 = np.array([0.5,-1.0,0.0,0.25,0.75])
h_prev2 = np.array([0.0,0.1,-0.1,0.2])

# Code starts here
def sigmoid(x):

    return np.where(x >= 0, 1/(1 + np.exp(-x)), np.exp(x)/(1 + np.exp(x)))

def as2d(x):

    x = np.asarray(x, dtype = float)
    dims = x.shape[-1]

    if x.ndim == 1:
        return x.reshape(1,dims) , True
    return x, False

def gru_forward_cell(x,h_prev):

    x = np.asarray(x, dtype = float)
    h_prev = np.asarray(h_prev, dtype = float)
    
    D = x.shape[-1]
    H = h_prev.shape[-1]

    x_2d, wasx_1d_x = as2d(x)
    h_prev2d, wash_1d_h = as2d(h_prev)
    was_1d = wasx_1d_x or wash_1d_h

    Wx = np.random.randn(D,H)
    Wh = np.random.randn(H,H)
    bh_11 = np.random.randn(H,)
    bh_12 = np.random.randn(H,)

    Wx2 = np.random.randn(D,H)
    Wh2 = np.random.randn(H,H)
    bh_21 = np.random.randn(H,)
    bh_22 = np.random.randn(H,)

    Wx3 = np.random.randn(D,H)
    Wh3 = np.random.randn(H,H)
    bh_31 = np.random.randn(H,)
    bh_32 = np.random.randn(H,)

    r = sigmoid(x_2d @ Wx + bh_11 + h_prev2d @ Wh + bh_12)
    z = sigmoid(x_2d @ Wx2 + bh_21 + h_prev2d @ Wh2 + bh_22)
    h_tiled = np.tanh(x_2d @ Wx3 + bh_31 + (r * h_prev2d) @ Wh3 + bh_32)
    ht = (1 - z) * h_prev2d + (z * h_tiled)

    if was_1d:
        return ht[0]

    return ht

