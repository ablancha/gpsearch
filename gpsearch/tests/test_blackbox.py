import numpy as np
from gpsearch import BlackBox, UniformInputs


def map_def1D(x):
    return np.sin(3*x) + x**2 - 0.7*x


def map_def2D(x):
    return np.linalg.norm(map_def1D(x))


def test_blackbox_shape_1D():
    n_init = 10
    domain = [[-1., 2.]]
    inputs = UniformInputs(domain)
    X = inputs.draw_samples(n_init, "lhs")
    Ys = BlackBox(map_def1D).evaluate(X)
    Yp = BlackBox(map_def1D).evaluate(X, parallel=True)
    np.testing.assert_allclose(Ys,Yp)


def test_blackbox_shape_2D():
    n_init = 10
    domain = [[-1., 2.], [-1., 2.]]
    inputs = UniformInputs(domain)
    X = inputs.draw_samples(n_init, "lhs")
    Ys = BlackBox(map_def2D).evaluate(X)
    Yp = BlackBox(map_def2D).evaluate(X, parallel=True)
    np.testing.assert_allclose(Ys,Yp)

