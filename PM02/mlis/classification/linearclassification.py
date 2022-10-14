import numpy as np


def h(w, X):
    """
    model/hypothesis
    :param w: weights/parameters
    :param X: data matrix with x_i^T in the ith row
    """
    return None  # <<<--- Replace this by your own result.


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
    w = w0  # initial weights
    ws = [w]
    for t in range(iters):  # for t = 1, ..., iters
        w = w  # <<<--- Replace this by your own result.
        ws.append(w)
    return np.array(ws)
