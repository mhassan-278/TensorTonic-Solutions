import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    if x.ndim == 3:
        # Shape: (C, H, W) -> (C,)
        return np.mean(x, axis=(1, 2))
    elif x.ndim == 4:
        # Shape: (N, C, H, W) -> (N, C)
        return np.mean(x, axis=(2, 3))
    else:
        raise ValueError("Input must be 3D (C,H,W) or 4D (N,C,H,W)")