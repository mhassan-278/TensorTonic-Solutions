import numpy as np

def adam_step(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Adam optimization step.

    Parameters:
    - param: scalar, list, or NumPy array (parameters)
    - grad: same shape as param (gradients)
    - m: first moment (same shape)
    - v: second moment (same shape)
    - t: timestep (1-based)
    - lr: learning rate
    - beta1, beta2: decay rates
    - eps: small constant for numerical stability

    Returns:
    - (param_new, m_new, v_new): all NumPy arrays
    """

    # Ensure NumPy arrays (float)
    param = np.asarray(param, dtype=np.float64)
    grad  = np.asarray(grad,  dtype=np.float64)
    m     = np.asarray(m,     dtype=np.float64)
    v     = np.asarray(v,     dtype=np.float64)

    # Update biased first and second moments
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # Bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    # Parameter update (elementwise)
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_new, v_new