def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Ensure inputs are float NumPy arrays
    x = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)

    D = x.shape[-1]
    H = h_prev.shape[-1]

    # Reshape to 2D if needed, track whether original was 1D
    x_2d, was_1d_x = _as2d(x, D)
    h_2d, was_1d_h = _as2d(h_prev, H)
    was_1d = was_1d_x or was_1d_h  # both should be consistent

    # Unpack parameters
    Wz, Uz, bz = params["Wz"], params["Uz"], params["bz"]
    Wr, Ur, br = params["Wr"], params["Ur"], params["br"]
    Wh, Uh, bh = params["Wh"], params["Uh"], params["bh"]

    # Update gate: z_t = σ(x_t @ Wz + h_{t-1} @ Uz + bz)
    z = _sigmoid(x_2d @ Wz + h_2d @ Uz + bz)
    # Reset gate: r_t = σ(x_t @ Wr + h_{t-1} @ Ur + br)
    r = _sigmoid(x_2d @ Wr + h_2d @ Ur + br)
    # Candidate hidden: h̃_t = tanh(x_t @ Wh + (r_t ⊙ h_{t-1}) @ Uh + bh)
    h_tilde = np.tanh(x_2d @ Wh + (r * h_2d) @ Uh + bh)
    # New hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
    h_t = (1 - z) * h_2d + z * h_tilde

    # Return to original dimensionality
    if was_1d:
        return h_t[0]
    return h_t