import numpy as np

def positional_encoding(seq_len, d_model, base=10000):
    
    pos = np.arange(seq_len, dtype=np.float64)[:, np.newaxis]
    dims = np.arange(d_model, dtype=np.float64)[np.newaxis, :]
    
    i = dims // 2   
    exponent = (2 * i) / d_model  
    angle = pos / (base ** exponent)
    
    result = np.where(dims % 2 == 0, np.sin(angle), np.cos(angle))
    return result