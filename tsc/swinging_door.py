import numpy as np


def swinging_door_sampling(x, y, eps):
    """
    Return signal compressed by the swinging-door algorithm.

    :param x: x-axis values of signal
    :param y: y-axis values of signal
    :param eps: error tolerance: small values yield low compression, high values loose compression
    :return: tuple (xc, yc) representing the compressed / sampled signal.
    """
    # otherwise, we get stuck in the while loop
    if eps <= 0:
        raise ValueError("eps must be > 0")
    
    # otherwise, x[i] with i = ~np.isnan(compressed) does not work
    if not isinstance(x, np.ndarray):
        x = np.array(x, copy=False)
    
    # array into which the selected values go
    y = np.asarray(y, "float")
    compressed = np.full_like(y, np.nan)
    
    # we always keep the first and the last value
    compressed[0] = y[0]
    compressed[-1] = y[-1]
    
    # k points to the last retained point
    k = 0
    
    # the currently observed maximal slope of the upper "door"
    upper_slope = -np.infty
    
    # the currently observed minimal slope of the lower "door"
    lower_slope = +np.infty
    
    # start investigating from the second point
    i = 1
    while i < len(x):
        # calculate current slopes between points k and i
        dx = x[i] - x[k]
        us = (y[i] - (y[k] + eps)) / dx
        ls = (y[i] - (y[k] - eps)) / dx
        
        if us > upper_slope:
            upper_slope = us
        
        if ls < lower_slope:
            lower_slope = ls
        
        # if the "door" has to be opened to much we ...
        if upper_slope >= lower_slope:
            # ... retain the last point
            compressed[i - 1] = y[i - 1]
            # and start over at the most recent point
            k = i - 1
            upper_slope = -np.infty
            lower_slope = +np.infty
        else:
            i += 1
    
    # return only retained points, i.e. positions where we have written non-nan values
    i = ~np.isnan(compressed)
    return x[i], compressed[i]


def compression_quality(x, y, eps):
    """
    Return a dictionary of metrics describing the result of compressing the signal (x,y)
    using the swinging-door algorithm with error tolerance eps.

    :param x: x-axis values of signal
    :param y: y-axis values of signal
    :param eps: error tolerance: small values yield low compression, high values loose compression
    :return: Dictionary of metrics.
              rmse ... root mean squared error between compressed signal and original signal
              mae ... mean absolute error between compressed signal and original signal
              max_err ... maximal absolute error between compressed signal and original signal
              epsilon ... the error tolerance passed
              rate ... The compression rate (len(y) / len(y_compressed))
    """
    xc, yc = swinging_door_sampling(x, y, eps)
    y_hat = np.interp(x, xc, yc)
    rmse = np.sqrt(np.mean((y - y_hat) ** 2))
    mae = np.mean(np.abs(y - y_hat))
    max_err = np.max(np.abs(y - y_hat))
    rate = len(y) / len(yc)
    return dict(
        rmse=rmse,
        mae=mae,
        max_err=max_err,
        epsilon=eps,
        rate=rate
    )
