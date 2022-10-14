import numpy as np

from mlis.arrays import asinput, aslabel
from mlis.kernels.kernels import sq_exp


def kernel_ridge(X, y, sigma, lam):
    """
    @param X: inputs, np.array of shape (m,n)
    @param y: labels, np.array of shape (m,)
    @param sigma: kernel bandwidth / length-scale
    @param lam: regularization parameter
    @return: A model h: X -> prediction
    """
    # make sure X and y are numpy arrays
    X, y = asinput(X), aslabel(y)
    return None  # <<<--- Replace this by your own result.
