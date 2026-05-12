import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5): 
    # Convert inputs to numpy arrays (float)
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    if x.ndim == 2:
        axes = (0,)
        # Reshape params to (1, D) for broadcasting
        gamma_expanded = gamma.reshape(1, -1)
        beta_expanded = beta.reshape(1, -1)
    elif x.ndim == 4:
        axes = (0, 2, 3)
        gamma_expanded = gamma.reshape(1, -1, 1, 1)
        beta_expanded = beta.reshape(1, -1, 1, 1)
    else:
        raise ValueError("Input must be 2D or 4D")

    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    y = gamma_expanded * x_hat + beta_expanded
    return y