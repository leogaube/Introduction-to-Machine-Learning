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
    α = np.zeros(X.shape[0])
    α = minimize(J, α, args=(X, y, sigma, lam), jac=dJ, method="bfgs").x
    return lambda Z: α @ sq_exp(X, Z, sigma)
    # <<<--- Replace this by your own result.


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
    K = sq_exp(X, X, sigma)

    objective = np.mean(np.log(1 + np.exp(-y * (α @ K))))
    regularization = lam * ((α.T @ K) @ α)

    return objective + regularization
    # <<<--- Replace this by your own result.


def dJ(α, X, y, sigma, lam):
    m, n = X.shape
    K = sq_exp(X, X, sigma)

    gradient = (-y @ K) / (1 + np.exp(y @ (α @ K)))
    regularization = 2 * lam * np.dot(K, α)

    return gradient / m + regularization
