import numpy as np


import numpy as np

def cosine_similarity(a, b):
    
    a = np.asarray(a)
    b = np.asarray(b)
    
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Handle zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)
