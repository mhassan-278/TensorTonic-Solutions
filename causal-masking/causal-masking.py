import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    
    scores = np.asarray(scores)
    masked = np.tril(np.ones_like(scores))
    return np.where(masked == 1,scores, mask_value)