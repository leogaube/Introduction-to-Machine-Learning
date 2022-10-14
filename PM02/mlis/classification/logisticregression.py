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
    # use logistic_loss(yhat, y)
    return None  # <<<--- Replace this by your own result.


def dJ(w, X, y):
    """
    logistic objective gradient
    @param w: weights/parameters
    @param X: data matrix with x_i^T in the ith row
    @param y: vector of labels
    @return: ∇J(w)
    """
    return None  # <<<--- Replace this by your own result.
