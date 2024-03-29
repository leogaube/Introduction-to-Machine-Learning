import numpy as np


def h(w, X):
    """
    model/hypothesis
    :param w: weights/parameters
    :param X: data matrix with x_i^T in the ith row
    """
    return X @ w  # <<<--- Replace this by your own result.


def J(w, X, y):
    """
    objective function
    :param w: weights/parameters
    :param X: data matrix with x_i^T in the ith row
    :param y: vector of labels
    """
    m, n = X.shape
    return 1 / m * sum((X @ w - y) ** 2)  # <<<--- Replace this by your own result.


def dJ(w, X, y):
    """
    gradient of objective function
    :param w: weights/parameters
    :param X: data matrix with x_i^T in the ith row
    :param y: vector of labels
    :return gradient of J with respect to w
    """
    m, n = X.shape
    return 2 / m * X.T @ (h(w, X) - y)  # <<<--- Replace this by your own result.


def gradientDescent(dJ, X, y, w0, eta, iters):
    """
    function that performs gradient descent
    :param dJ: function which calculates the gradient of the objective J
    :param X: data matrix with x_i^T in the ith row
    :param y: vector of labels
    :param w0: initial parameters
    :param eta: learning rate η
    :param iters: number of gradient descent steps
    :return list (history) of all calculated parameters
    """
    m, n = X.shape
    w = w0  # initial weights
    ws = [w]
    for t in range(iters):  # for t = 1, ..., iters
        w = w - eta * dJ(w, X, y)  # <<<--- Replace this by your own result.
        ws.append(w)
    return np.array(ws)


def least_squares(X, y):
    """
    :param X: data matrix with x_i^T in the ith row
    :param y: vector of labels
    :return: The analytic solution to least squares
    """
    return np.linalg.solve(X.T @ X, X.T @ y)  # <<<--- Replace this by your own result.
