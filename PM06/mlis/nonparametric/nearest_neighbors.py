import numpy as np

from mlis.arrays import asinput, aslabel


def knn_classify(X, y, Z, k, dist):
    """
    k nearest neighbor classification
    @param X: m by n matrix of inputs
    @param y: m by 1 matrix of labels
    @param Z: mz by n matrix of inputs
    @param k: how many neighbors are considered?
    @param dist: A distance function from scipy.spatial.distance.cdist
    @return: the label prediction by k nearest neighbor classification
    """
    X, Z, y = asinput(X), asinput(Z), aslabel(y)
    m, n = X.shape

    distances = dist(Z, X)
    if k < m:
        idx = np.argpartition(distances, k, axis=1)[:, :k]
    else:
        idx = np.indices((m, m))[1]  # <-- column indices
    return np.mean(y[idx], axis=1)


# <<<--- Replace this by your own result.
