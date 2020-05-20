import sys
sys.path.append('../../')

import numpy as np
import gpsearch
from gpsearch.examples import Oscillator, Noise
from gpsearch.core import BlackBox, GaussianInputs, \
                          OptimalDesign, Likelihood, Q, QInt


def map_def(theta, oscil):
    u, t = oscil.solve(theta)
    mean_disp = np.mean(u[:,0])
    return mean_disp


def main():

    ndim = 2
    np.random.seed(2)

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

    X = inputs.draw_samples(15, "lhs")
    Y = myMap.evaluate(X)
    o = OptimalDesign(X, Y, myMap, inputs, normalize_Y=True)
    likelihood = Likelihood(o.model, o.inputs)
    x_new = np.atleast_2d([1.0,2.0])

    qcrit = Q(o.model, o.inputs, likelihood=likelihood)
    print(qcrit.evaluate(x_new))
    print(qcrit.jacobian(x_new))

    qcrit = QInt(o.model, o.inputs, ngrid=250, likelihood=likelihood)
    print(qcrit.evaluate(x_new))
    print(qcrit.jacobian(x_new))


if __name__ == "__main__":
    main()
