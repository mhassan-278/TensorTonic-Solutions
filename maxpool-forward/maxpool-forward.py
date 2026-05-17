def maxpool_forward(X, pool_size, stride):
    # Convert to list of lists if X is a numpy array
    if hasattr(X, 'tolist'):
        X = X.tolist()
    
    H = len(X)
    W = len(X[0]) if H > 0 else 0
    
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    out = []
    for i in range(H_out):
        row = []
        for j in range(W_out):
            start_row = i * stride
            start_col = j * stride
            # Compute max efficiently with a double loop
            max_val = X[start_row][start_col]
            for a in range(pool_size):
                for b in range(pool_size):
                    val = X[start_row + a][start_col + b]
                    if val > max_val:
                        max_val = val
            row.append(max_val)
        out.append(row)
    return out