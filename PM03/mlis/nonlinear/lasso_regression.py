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
    y_hat = poly_feat(X, degree) @ w
    regularization = lam * np.sum(np.abs(w))

    return np.mean((y_hat - y) ** 2 + regularization, axis=0)


# <<<--- Replace this by your own result.


def dJ(w, X, y, degree, lam):
    """
    gradient of objective function
    :param w: weights/parameters
    :param X: data matrix with x_i^T in the ith row
    :param y: vector of labels
    :param lam: regularization (>=0)
    :param degree: polynomial degree of feature function
    """
    poly_X = poly_feat(X, degree)

    y_hat = poly_X @ w
    regularization = lam * np.sign(w)

    return 2 / len(y) * poly_X.T @ (y_hat - y) + regularization


# <<<--- Replace this by your own result.
