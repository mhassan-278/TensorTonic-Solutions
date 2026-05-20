import numpy as np

def rnn_step_backward(dh, cache):
    """
    Backward pass for a single RNN time step (vanilla RNN with tanh).

    Args:
        dh: Upstream gradient of loss with respect to current hidden state,
            shape (H,).
        cache: List containing cached values from forward pass:
            [x_t, h_prev, h_t, W, U, b]

    Returns:
        Tuple of gradients:
            dx_t: gradient with respect to x_t, shape (D,)
            dh_prev: gradient with respect to h_prev, shape (H,)
            dW: gradient with respect to W, shape (H, D)
            dU: gradient with respect to U, shape (H, H)
            db: gradient with respect to b, shape (H,)
    """
    # Convert all cached values to NumPy arrays (handles lists and arrays)
    x_t = np.array(cache[0])
    h_prev = np.array(cache[1])
    h_t = np.array(cache[2])
    W = np.array(cache[3])
    U = np.array(cache[4])
    b = np.array(cache[5])
    dh = np.array(dh)

    # Gradient through tanh: dz = dh * (1 - h_t^2)
    dz = dh * (1 - h_t ** 2)   # shape (H,)

    # Gradients for weight matrices and bias
    dW = np.outer(dz, x_t)     # shape (H, D)
    dU = np.outer(dz, h_prev)  # shape (H, H)
    db = dz.copy()             # shape (H,)

    # Gradients for inputs
    dx_t = W.T @ dz            # shape (D,)
    dh_prev = U.T @ dz         # shape (H,)

    return dx_t, dh_prev, dW, dU, db