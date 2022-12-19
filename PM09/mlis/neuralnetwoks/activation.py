def ReLu(x):
    """
    Vectorized version of the ReLu action function
    @param x: inputs
    @return:  max(0,x)
    """
    return (x > 0) * x


# <<<--- Replace this by your own result.
