import numpy as np

from mlis.arrays import asinput


def sq_dist(X, Z):
    """
    Compute squared Euclidean distance between each pair of the two collections of inputs.
    @param X: mx by n matrix of inputs
    @param Z: mz by n matrix of inputs
    @return: The squared Euclidean distance between each pair
    """
    # make sure X and Z are numpy arrays
    X, Z = asinput(X), asinput(Z)
    return (X ** 2).sum(axis=1)[:, None] + (Z ** 2).sum(axis=1) - np.dot(2 * X, Z.T)
    # <<<--- Replace this by your own result.


def sq_exp(X, Z, sigma):
    """
    Computes the kernel matrix K of the two collections of inputs.
    @param X: mx by n matrix of inputs
    @param Z: mz by n matrix of inputs
    @param sigma: kernel bandwidth / length-scale
    @return: The kernel matrix K
    """
    assert sigma > 0
    return np.exp(-sq_dist(X, Z) / (2 * sigma ** 2))
    # <<<--- Replace this by your own result.
