def xavier_initialization(W, fan_in, fan_out):

    L = np.sqrt(6 / (fan_in + fan_out))
    W = np.asarray(W)
    W = W * 2*L - L

    return W