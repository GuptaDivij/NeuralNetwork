import numpy as np

# Implementing Mean Squared Error (MSE) for regression
def mse_loss(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)

    # diff = y_pred - y_true
    diff = y_pred - y_true

    # loss = mean(diff^2)
    loss = np.mean(diff * diff).astype(np.float32)

    # d/dy_pred mean(diff^2) = 2*diff / N
    grad = (2.0 / diff.size) * diff
    return loss, grad.astype(np.float32)
