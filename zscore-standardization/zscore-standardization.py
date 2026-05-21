import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):

    X = np.array(X)
    mean = X.mean(axis = axis,keepdims= True)
    std = X.std(axis = axis, keepdims = True)

    return (X - mean) / (std + eps)