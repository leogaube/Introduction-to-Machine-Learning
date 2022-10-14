import numpy as np

from mlis.nonlinear.features import poly_feat


def ls_poly(X, y, degree):
    """
    Polynomial Least Squared
    @param X: inputs, np.array of shape (m,n)
    @param y: labels, np.array of shape (m,)
    @param degree: polynomial degree (>=0)
    @return: A function X -> <ϕ(X),w>
    """
    return lambda X: None  # <<<--- Replace this by your own result.


def ridge_poly(X, y, degree, lam):
    """
    Polynomial Ridge Regression
    @param X: inputs, np.array of shape (m,n)
    @param y: labels, np.array of shape (m,)
    @param degree: polynomial degree (>=0)
    @param lam: lambda (λ >=0)
    @return: A function X -> <ϕ(X),w>
    """
    return lambda X: None  # <<<--- Replace this by your own result.
