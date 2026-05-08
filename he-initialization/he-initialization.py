def he_initialization(W, fan_in):

    L = np.sqrt(6/fan_in)
    W = np.asarray(W)
    W = W * 2*L - L

    return W