import autograd.numpy as np

from mlis.arrays import asinput
from mlis.neuralnetwoks.activation import ReLu


def init(layer_sizes, scale, rng):
    """Build a list of randomly initialized (U, c) tuples, one for each layer."""
    params = []
    for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:]):
        U = rng.randn(insize, outsize) * scale
        c = rng.randn(outsize) * scale
        params.append((U, c))
    return params


def predict(X, params, activation=ReLu):
    """
    𝑘-layer hypothesis function
    @param X: inputs of shape (m,n)
    @param params: The list `params` contains the tuples $(U_i, c_i)$ i.e. `[ (U_1, c_1), ..., (U_k, c_k) ]`.
    @param activation: The activation function
    @return: neural network prediction
    """
    z = asinput(X)
    for W, b in params[:-1]:
        outputs = np.dot(z, W) + b
        z = activation(outputs)
    # no activation on the last layer
    W, b = params[-1]
    return np.dot(z, W) + b
    return None  # <<<--- Replace this by your own result.
