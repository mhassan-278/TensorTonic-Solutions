import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    diff = y_true - y_pred

    loss = np.where(
    np.abs(diff) <= delta,
    (0.5 * diff ** 2),
    delta * (np.abs(diff) - (0.5 * delta))
)
    
    return np.mean(loss)
