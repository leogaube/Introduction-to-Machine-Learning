import numpy as np

from mlis.arrays import asinput


def bootstrap(X, y, rng):
    """
    create a new bootstrap data set from (X,y) with the same number of samples as the original data set.
    @param X: m by 1 array of inputs
    @param y: m by 1 array of labels
    @param rng: numpy RandomState instance.
    @return: A bootstrap data set
    """
    # rng.choice might be good function to use here.
    return X, y  # <<<--- Replace this by your own result.


def fit_ensemble(X, y, base_learner, T, rng):
    """
    fit an ensemble to the data
    @param X: m by 1 array of inputs
    @param y: m by 1 array of labels
    @param base_learner: The base learner
    @param T: Ensemble size / number of base learner
    @param rng: numpy RandomState instance.
    @return: A collection of fitted base_learner
    """
    return []  # <<<--- Replace this by your own result.


def ensemble_predict(X, ensemble):
    """
    Make predictions for X using the ensemble
    @param X: m by 1 array of inputs
    @param ensemble: A collection of fitted base_learner
    @return: predictions for X using the ensemble
    """
    X = asinput(X)
    return None  # <<<--- Replace this by your own result.
