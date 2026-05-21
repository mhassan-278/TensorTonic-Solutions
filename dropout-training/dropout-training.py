import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.asarray(x)
    if rng is None:
        rand = np.random.random(x.shape)
    else:
        rand = rng.random(x.shape)
    
    scale = 1.0 / (1.0 - p)
    # mask is True where we keep the element (random >= p)
    mask = rand >= p
    
    # Apply dropout and scale
    output = x * mask * scale
    dropout_pattern = mask * scale
    
    return output, dropout_pattern