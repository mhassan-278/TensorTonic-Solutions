import math

def elu(x, alpha):
    
    return [i if i>0 else (alpha * (math.exp(i) - 1)) for i in x]