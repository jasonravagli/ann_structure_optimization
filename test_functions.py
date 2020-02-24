def himmelblau(X):
    """
    http://en.wikipedia.org/wiki/Himmelblau%27s_function
    This function has four local minima where the value of the function is 0.
    """
    x = X[0]
    y = X[1]
    a = x*x + y - 11
    b = x + y*y - 7
    return a*a + b*b
