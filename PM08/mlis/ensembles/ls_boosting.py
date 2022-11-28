import numpy as np
from sklearn.tree import DecisionTreeRegressor

from mlis.arrays import asinput, aslabel


def stump_fit(X, y):
    # sklearn expects X to be a matrix, so we reshape it to a mx1 matrix.
    h = DecisionTreeRegressor(max_depth=1)
    h.fit(asinput(X), aslabel(y))
    # sklearn expects X to be a matrix, so we reshape it to a mx1 matrix.
    predict = lambda Z: h.predict(asinput(Z))
    return predict


def ls_boosting_fit(X, y, T):
    """
    Learn a least squares boosting ensemble
    @param X: m by 1 array of inputs
    @param y: m by 1 array of labels
    @param T: Ensemble size / number of base learner
    @return: A (ensemble) model H: X -> yhat
    """
    h_functions = []
    H = lambda X: sum([h(X) for h in h_functions])
    for _ in range(T):
        h = stump_fit(X, y - H(X))
        h_functions.append(h)
    return H


# <<<--- Replace this by your own result.
