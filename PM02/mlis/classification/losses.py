import numpy as np


def logistic_loss(yhat, y):
    """
    logistic loss function
    :param yhat: model prediction (scalar or array)
    :param y: actual label  (scalar or array)
    :return: logistic loss
    """
    return np.log(1 + np.exp(-y * yhat))  # <<<--- Replace this by your own result.


def zero_one_loss(yhat, y):
    """
    zero_one_loss function
    :param yhat: model prediction (scalar or array)
    :param y: actual label  (scalar or array)
    :return: logistic loss
    """
    return (np.sign(yhat * y) == -1).astype(np.float64)
    # <<<--- Replace this by your own result.


def hinge_loss(yhat, y):
    """
    hinge loss function
    :param yhat: model prediction (scalar or array)
    :param y: actual label  (scalar or array)
    :return: hinge loss
    """

    return np.maximum(np.zeros((y.shape)), 1 - y * yhat)
    # <<<--- Replace this by your own result.


def logistic(z):
    """
    the logistic function
    :param z: model prediction
    :return: z squashed to [0,1]
    """
    return 1 / (1 + np.exp(-z))  # <<<--- Replace this by your own result.
