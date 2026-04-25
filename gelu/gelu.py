def gelu(x):

    x = np.asarray(x, dtype= float)
    erf_vec = np.vectorize(math.erf)

    return 0.5 * x * (1 + erf_vec(x/np.sqrt(2)))