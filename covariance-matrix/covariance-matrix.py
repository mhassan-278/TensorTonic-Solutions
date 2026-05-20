import numpy as np

def covariance_matrix(X):
    
    X = np.asarray(X)
    
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    
    N, D = X.shape
    
    # Step 1: Center the data
    mean = np.mean(X, axis=0)           # shape (D,)
    X_centered = X - mean               # shape (N, D)
    
    # Step 2: Compute covariance matrix using matrix multiplication
    # Cov = (1/(N-1)) * X_centered.T @ X_centered
    cov = (X_centered.T @ X_centered) / (N - 1)
    
    return cov