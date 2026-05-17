import numpy as np

def conv2d(x, W, b):

    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape
    
    windows = np.lib.stride_tricks.sliding_window_view(x, (KH, KW), axis=(2, 3))
    y = np.einsum('nchwkl, ockl -> nohw', windows, W, optimize=True)
    y += b.reshape(1, -1, 1, 1)

    return y
