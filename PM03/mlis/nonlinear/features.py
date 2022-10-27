from itertools import combinations_with_replacement

import numpy as np

from mlis.arrays import asinput


def poly_feat(X, degree):
    """
    Polynomial Feature Function
    @param X: inputs, np.array of shape (m,n)
    @param degree: polynomial degree (>=0)
    @return: A np.array [X^0, ..., X^degree ]
    """
    assert degree >= 0
    # make sure X is a numpy array
    X = asinput(X)
    return np.squeeze([X ** d for d in range(degree + 1)]).T
    # <<<--- Replace this by your own result.
