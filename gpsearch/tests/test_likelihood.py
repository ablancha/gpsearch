import numpy as np
from matplotlib import pyplot as plt
from GPy.models import GradientChecker
from gpsearch.examples import Oscillator, Noise
from gpsearch.core import (BlackBox, GaussianInputs,
                           OptimalDesign, Likelihood)


def map_def(theta, oscil):
    u, t = oscil.solve(theta)
    mean_disp = np.mean(u[:,0])
    return mean_disp


ndim = 2
tf = 25
nsteps = 1000
u_init = [0, 0]
noise = Noise([0, tf])
oscil = Oscillator(noise, tf, nsteps, u_init)
myMap = BlackBox(map_def, args=(oscil,))

lam = noise.get_eigenvalues(ndim)
mean = np.zeros(ndim)
cov = np.diag(lam)

domain = [ [-a, a] for a in 6.0*np.sqrt(np.diag(cov)) ] 
inputs = GaussianInputs(domain, mean, cov)


def test_likelihood_gradients_nominal():
    X = inputs.draw_samples(100, "lhs")
    Y = myMap.evaluate(X, parallel=True)
    o = OptimalDesign(X, Y, myMap, inputs, normalize_Y=True)
    x_new = np.atleast_2d([1.0,2.0])
    kwargs_gmm = dict(n_components=4, covariance_type="full")
    likelihood = Likelihood(o.model, o.inputs, "nominal", 
                            kwargs_gmm=kwargs_gmm)
    gm = GradientChecker(lambda x: likelihood.evaluate(x),
                         lambda x: likelihood.jacobian(x), 
                         x_new, 'x')
    assert(gm.checkgrad())


def test_likelihood_gradients_importance():
    X = inputs.draw_samples(100, "lhs")
    Y = myMap.evaluate(X, parallel=True)
    o = OptimalDesign(X, Y, myMap, inputs, normalize_Y=True)
    x_new = np.atleast_2d([1.0,2.0])
    kwargs_gmm = dict(n_components=4, covariance_type="full")
    likelihood = Likelihood(o.model, o.inputs, "importance",
                            kwargs_gmm=kwargs_gmm)
    gm = GradientChecker(lambda x: likelihood.evaluate(x),
                         lambda x: likelihood.jacobian(x),
                         x_new, 'x')
    assert(gm.checkgrad())




