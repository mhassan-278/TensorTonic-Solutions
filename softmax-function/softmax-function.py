def softmax(x):
    
    x = np.asarray(x)
    exp_x = np.exp(x - x.max(axis= -1, keepdims= True))
    return exp_x /  np.sum(exp_x,axis= -1, keepdims= True)