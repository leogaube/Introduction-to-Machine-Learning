import numpy as np
from mlis.classification.linearclassification import h
from mlis.classification.losses import logistic_loss


def J(w, X, y):
    """
    logistic objective function
    @param w: weights/parameters
    @param X: data matrix with x_i^T in the ith row
    @param y: vector of labels
    @return: J(w)
    """
    yhat = h(w, X)
    return (
        1 / yhat.shape[0] * np.sum(logistic_loss(yhat, y))
    )  # <<<--- Replace this by your own result.


def dJ(w, X, y):
    """
    logistic objective gradient
    @param w: weights/parameters
    @param X: data matrix with x_i^T in the ith row
    @param y: vector of labels
    @return: ∇J(w)
    """

    # unoptimized for loop
    # m, n = X.shape
    # sum = 0
    # for i in range(m):
    #     sum += -y[i] * X[i] / (1 + np.exp(y[i] * h(w, X[i])))

    num = -y[:, np.newaxis] * X
    den = 1 + np.exp(y[:, np.newaxis] * h(w, X)[:, np.newaxis])

    return np.mean(num / den, axis=0)  # <<<--- Replace this by your own result.
