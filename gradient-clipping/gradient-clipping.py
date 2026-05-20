import numpy as np

def clip_gradients(g, max_norm):
    g = np.asarray(g)
    # If max_norm is not positive, no clipping (edge case)
    if max_norm <= 0:
        return g
    
    norm = np.sqrt(np.sum(g ** 2))
    # If norm is zero, nothing to clip
    if norm == 0:
        return g
    
    # Apply clipping only if norm exceeds threshold
    if norm > max_norm:
        return g * (max_norm / norm)
    return g