import numpy as np

from mlis.arrays import asinput, aslabel


def findSplitIdx(X, splits):
    """
    @param X: m by 1 array of inputs
    @param splits: A set of split points. `splits = [20, 60]` defines 3 partitions (-oo, 20], (20, 60] and (60, oo).
    @return: The partition index for each x in X, starting at 0.
    """
    # make sure X is a np.array of shape (m,n)
    X = asinput(X)
    # sorted and no duplicates
    splits = np.unique(np.sort(splits))
    return None  # <<<--- Replace this by your own result.


def predict(X, yhats, splits):
    """
    Make predictions for inputs X
    @param X: m by 1 array of inputs
    @param yhats: m by 1 array of constant predictions for each partition
    @param splits: A set of split points. `splits = [20, 60]` defines 3 partitions (-oo, 20], (20, 60] and (60, oo).
    @return: An array of predictions, one for each x in X.
    """
    # P partitions are defined with P-1 split points
    assert len(yhats) == len(splits) + 1, f'{len(yhats)} == {len(splits) + 1}'
    # make sure X and yhat are numpy arrays
    X, yhats = asinput(X), aslabel(yhats)
    return None  # <<<--- Replace this by your own result.


def fit_sq_loss(X, y, splits):
    """
    Calculates the optimal yhat for each partition using the square loss.
    @param X: m by 1 array of inputs
    @param y: m by 1 array of labels
    @param splits: A set of split points. `splits = [20, 60]` defines 3 partitions (-oo, 20], (20, 60] and (60, oo).
    @return: yhat
    """
    X, y = asinput(X), aslabel(y)
    return None  # <<<--- Replace this by your own result.


def J(X, y, splits):
    """
    The objective function which tells us how good the split is using the square loss.
    @param X: m by 1 array of inputs
    @param y: m by 1 array of labels
    @param splits: A set of split points. `splits = [20, 60]` defines 3 partitions (-oo, 20], (20, 60] and (60, oo).
    @return: J(split)
    """
    X, y = asinput(X), aslabel(y)
    return None  # <<<--- Replace this by your own result.


def find_next_split(X, y, splits):
    """
    Find the next split in a greedy way
    @param X: m by 1 array of inputs
    @param y: m by 1 array of labels
    @param splits: The splits so far.
    @return: The next best split which is in the middle between two points in X
    If no split reduces the objective $J$, return `None`
    """
    return None  # <<<--- Replace this by your own result.


def fit_tree(X, y, max_splits):
    """
    Fits a regression tree with square loss.
    @param X: m by 1 array of inputs
    @param y: m by 1 array of labels
    @param max_splits: split until find_next_split returns None, but not more than max_splits
    @return: A model h: X -> R
    """
    assert max_splits >= 1
    return None  # <<<--- Replace this by your own result.
