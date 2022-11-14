import cvxopt
import numpy as np

from mlis.arrays import asinput, aslabel


# wrapper around cvxopt
def quadprog(P, q, G, h, A, b):
    sol = cvxopt.solvers.qp(
        cvxopt.matrix(P, tc="d"),
        cvxopt.matrix(q, tc="d"),
        cvxopt.matrix(G, tc="d"),
        cvxopt.matrix(h, tc="d"),
        cvxopt.matrix(A, tc="d"),
        cvxopt.matrix(b, tc="d"),
    )
    return np.ravel(sol["x"])


def lagrange_multipliers(y, C, K):
    """
    @param y: labels, np.array of shape (m,)
    @param C: svm regularizer
    @param K: Kernel matrix, np.array of shape (m,m)
    @return: alpha
    """
    y = aslabel(y).astype(float)
    m = K.shape[0]

    # --- Replace this by your own result.
    P = (y * y.T) * K
    q = -1 * np.ones((m))
    G = np.concatenate((-1 * np.identity(m), np.identity(m)))
    h = np.concatenate((np.zeros(m), C * np.ones(m)))
    A = y.reshape((1, y.shape[0]))
    b = 0.0
    alpha = quadprog(P, q, G, h, A, b)
    return alpha


def bias(y, C, alpha, K):
    """
    @param y: labels, np.array of shape (m,)
    @param C: svm regularizer
    @param alpha: kernel parameters, np.array of shape (m,)
    @param K: Kernel matrix, np.array of shape (m,m)
    @return: b, float
    """
    # for stability, use not a single, but all alphas which fulfill
    # the KKT criteria and then average the result
    idx = (alpha > 0) & (alpha < C)
    biases = y[idx] - (K[idx, :][:, idx] @ (alpha[idx] * y[idx]))

    return np.mean(biases)  # <<<--- Replace this by your own result.


def svm_fit(X, y, C, kernel):
    """
    @param X: inputs, np.array of shape (m,n)
    @param y: labels, np.array of shape (m,)
    @param C:
    @param kernel: kernel function X x X -> R
    @return: A model h: X -> yhat
    """
    X, y = asinput(X), aslabel(y)
    K = kernel(X, X)  # obtain kernel matrix
    alpha = lagrange_multipliers(y, C, K)  # obtain lagrange multiplier
    b = bias(y, C, alpha, K)  # calculate bias

    # select support vectors.
    sidx = alpha > 1e-5
    SVs = X[sidx, :]

    # SVM classification function
    def h(Z):
        K = kernel(SVs, Z)  # compute kernel matrix
        beta = np.diag(alpha[sidx] * y[sidx])
        return np.sum(np.dot(beta, K), axis=0) + b

    return h
