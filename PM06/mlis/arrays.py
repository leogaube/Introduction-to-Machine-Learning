import numpy as np


def asinput(X):
    """
    make sure that the data is a numpy array of shape (m,n)
    """
    X = np.asarray(X)
    assert X.ndim in {1, 2}
    X = X.reshape((-1, 1)) if X.ndim == 1 else X
    X.flags.writeable = False
    return X


def aslabel(y):
    """
    make sure that the labels are a numpy array
    """
    y = np.asarray(y)
    y.flags.writeable = False
    return y
