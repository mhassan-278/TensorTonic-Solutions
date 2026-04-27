import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    cos = (x1 @ x2.T) / (np.sqrt(np.sum(x1**2)) * np.sqrt(np.sum(x2**2)))
    return (1-cos) if label == 1 else np.maximum(0,cos - margin)
