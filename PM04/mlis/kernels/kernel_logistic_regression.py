import numpy as np
from scipy.optimize import minimize

from mlis.kernels.kernels import sq_exp


def kernel_lr(X, y, sigma, lam):
    """
    kernel ridge regression using the scipy optimizer gradient descent
    @param X: inputs, np.array of shape (m,n)
    @param y: labels, np.array of shape (m,)
    @param sigma: kernel bandwidth / length-scale
    @param lam: regularization parameter
    @return: A model h: X -> prediction
    """
    α = np.zeros(X.shape[0], )
    α = minimize(J, α, args=(X, y, sigma, lam), jac=dJ, method='bfgs').x
    return None  # <<<--- Replace this by your own result.


def J(α, X, y, sigma, lam):
    """
    Kernel logistic objective function
    @param α: kernel weights / parameters
    @param X: data matrix with x_i^T in the ith row
    @param y: vector of labels
    @param sigma: kernel bandwidth / length-scale
    @param lam: regularization parameter
    @return: J(w)
    """

    return None  # <<<--- Replace this by your own result.


def dJ(α, X, y, sigma, lam):
    """
    Gradient of the kernel logistic objective function
    @param α: kernel weights / parameters
    @param X: data matrix with x_i^T in the ith row
    @param y: vector of labels
    @param sigma: kernel bandwidth / length-scale
    @param lam: regularization parameter
    @return:  ∇J(w)
    """

    return None  # <<<--- Replace this by your own result.
