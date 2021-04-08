import sys
sys.path.append('../../')

import numpy as np
import gpsearch
from gpsearch import BlackBox, UniformInputs, OptimalDesign


def map_def1D(x):
    return np.sin(3*x) + x**2 - 0.7*x


def map_def2D(x):
    return np.linalg.norm(map_def1D(x))


def main():

    np.random.seed(2)
    n_init = 10

    domain = [[-1., 2.]]
    inputs = UniformInputs(domain)
    X = inputs.draw_samples(n_init, "lhs")
    Ys = BlackBox(map_def1D).evaluate(X)
    Yp = BlackBox(map_def1D).evaluate(X, parallel=True)
    OptimalDesign(X, Ys, BlackBox(map_def1D), inputs)
    OptimalDesign(X, Yp, BlackBox(map_def1D), inputs)
    print(X, Ys, Yp)

    domain = [[-1., 2.], [-1., 2.]]
    inputs = UniformInputs(domain)
    X = inputs.draw_samples(n_init, "lhs")
    Ys = BlackBox(map_def2D).evaluate(X)
    Yp = BlackBox(map_def2D).evaluate(X, parallel=True)
    OptimalDesign(X, Ys, BlackBox(map_def2D), inputs)
    OptimalDesign(X, Yp, BlackBox(map_def2D), inputs)
    print(X, Ys, Yp)


if __name__ == "__main__":
    main()
