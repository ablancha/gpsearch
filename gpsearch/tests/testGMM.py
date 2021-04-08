import sys
sys.path.append('../../')

import numpy as np
import gpsearch
from gpsearch.examples import Oscillator, Noise
from gpsearch.core import (BlackBox, GaussianInputs, UniformInputs,
                           OptimalDesign, Likelihood, jacobian_fdiff)
from matplotlib import pyplot as plt


def map_def(theta, oscil):
    u, t = oscil.solve(theta)
    mean_disp = np.mean(u[:,0])
    return mean_disp


def main():

    ndim = 2
    np.random.seed(3)

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
   #inputs = UniformInputs(domain)

    kwargs_gmm = dict(n_components=4, covariance_type="spherical")

    X = inputs.draw_samples(100, "lhs")
    Y = myMap.evaluate(X, parallel=True)
    o = OptimalDesign(X, Y, myMap, inputs, normalize_Y=True)
    likelihood = Likelihood(o.model, o.inputs, "nominal", kwargs_gmm=kwargs_gmm)

    x_new = np.atleast_2d([1.0,2.0])
    gmm_y = likelihood.evaluate(x_new)
    print(jacobian_fdiff(likelihood, x_new))
    print(likelihood.jacobian(x_new))

    from GPy.models import GradientChecker
    gm = GradientChecker(lambda x: likelihood.evaluate(x),
                         lambda x: likelihood.jacobian(x), 
                         x_new, 'x')
    assert(gm.checkgrad())

    pts = inputs.draw_samples(n_samples=100, sample_method="grd")
    gmm_y = likelihood.evaluate(pts).flatten()
    pix = likelihood._evaluate_raw(pts).flatten()

    fig = plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sc = plt.scatter(pts[:,0], pts[:,1], c=pix)
    plt.colorbar(sc)
    plt.title(r"$f_x/f_y$")
    plt.subplot(1,2,2)
    sc = plt.scatter(pts[:,0], pts[:,1], c=gmm_y)
    plt.colorbar(sc)
    plt.title("GMM fit")
    plt.show()


if __name__ == "__main__":
    main()
