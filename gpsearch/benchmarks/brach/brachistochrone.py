import numpy as np
from scipy.optimize import newton


def cycloid(x2, y2, N=100):

    def fun(theta):
        """Find theta2 from (x2, y2) numerically"""
        return y2/x2 - (1-np.cos(theta))/(theta-np.sin(theta))

    theta2 = newton(fun, np.pi/2)
    R = y2 / (1 - np.cos(theta2))
    theta = np.linspace(0, theta2, N)
    x = R * (theta - np.sin(theta))
    y = R * (1 - np.cos(theta))

    x[-1] = x2
    y[-1] = y2

    return x, y


def travel_time(x, y):
    x, y = x.flatten(), y.flatten()
    g = 9.81
    slopes = np.diff(y)/np.diff(x)
    times = np.diff(x) * np.sqrt(2 * (1+slopes**2) / g) \
            / (np.sqrt(y[:-1]) + np.sqrt(y[1:]))
    return np.sum(times)


