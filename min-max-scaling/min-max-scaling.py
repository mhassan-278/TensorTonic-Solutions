import numpy as np

def min_max_scaling(data):

    arr = np.asarray(data, dtype=float)
    col_min = arr.min(axis=0)
    col_max = arr.max(axis=0)
    col_range = col_max - col_min
    
    # Use np.divide with where clause: where range != 0, do (arr - min)/range;
    # otherwise fill with 0. This avoids NaN from 0/0.
    scaled = np.divide(
        arr - col_min,
        col_range,
        out=np.zeros_like(arr),
        where=(col_range != 0)
    )
    return scaled.tolist()