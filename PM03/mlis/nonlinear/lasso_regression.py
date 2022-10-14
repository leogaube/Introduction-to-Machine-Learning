import numpy as np

from mlis.nonlinear.features import poly_feat


def J(w, X, y, degree, lam):
    """
    objective function
    :param w: weights/parameters
    :param X: data matrix with x_i^T in the ith row
    :param y: vector of labels
    :param degree: polynomial degree of feature function
    :param lam: regularization (>=0)
    """
    return None  # <<<--- Replace this by your own result.


def dJ(w, X, y, degree, lam):
    """
    gradient of objective function
    :param w: weights/parameters
    :param X: data matrix with x_i^T in the ith row
    :param y: vector of labels
    :param lam: regularization (>=0)
    :param degree: polynomial degree of feature function
    """
    return None  # <<<--- Replace this by your own result.
