import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid function.

    Works with:
    - Scalars
    - Python lists
    - NumPy arrays

    Returns:
    - NumPy array of floats
    """
    x = np.asarray(x, dtype=np.float64)  # ensures array + float type
    return 1 / (1 + np.exp(-x))