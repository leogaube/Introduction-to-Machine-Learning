import numpy as np
from mlis.classification.linearclassification import h
from mlis.classification.losses import hinge_loss


def J(w, X, y):
    """
    svm objective function
    :param w: weights/parameters
    :param X: data matrix with x_i^T in the ith row
    :param y: vector of labels
    """
    # print(
    #     np.array_equal(
    #         hinge_loss(h(w, X), y), (np.where(1 - y * h(w, X) > 0, 1 - y * h(w, X), 0))
    #     )
    # )
    # return hinge_loss(h(w, X), y)          <-- non-scalar?!
    return np.mean(np.where(1 - y * h(w, X) > 0, 1 - y * h(w, X), 0))
    # <<<--- Replace this by your own result.


def dJ(w, X, y):
    """
    svm objective gradient
    :param w: weights/parameters
    :param X: data matrix with x_i^T in the ith row
    :param y: vector of labels
    """
    return np.sum((-X * y[:, np.newaxis])[(y * h(w, X)) <= 1], axis=0) / len(y)
    # <<<--- Replace this by your own result.
