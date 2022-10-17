import numpy as np

from mlis.regression.linearregression import h


def J(w, X, y):
    """
    absolute loss objective function
    :param w: weights/parameters
    :param X: data matrix with x_i^T in the ith row
    :param y: vector of labels
    :return absolute loss objective function
    """
    m, n = X.shape
    return 1 / m * np.sum(np.abs(X @ w - y))  # <<<--- Replace this by your own result.


def dJ(w, X, y):
    """
    gradient of absolute loss objective function
    :param w: weights/parameters
    :param X: data matrix with x_i^T in the ith row
    :param y: vector of labels
    :return gradient of J with respect to w
    """
    m, n = X.shape
    return 1 / m * X.T @ np.sign(h(w, X) - y)  # <<<--- Replace this by your own result.
