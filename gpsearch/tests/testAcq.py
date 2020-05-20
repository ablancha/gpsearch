import sys
sys.path.append('../../')

import numpy as np
import gpsearch
from gpsearch.examples import Oscillator, Noise
from gpsearch.core import BlackBox, GaussianInputs, \
                          OptimalDesign, Likelihood, Q, QInt, \
                          IVR, IVRInt, IVR_LW, IVR_LWInt, \
                          AcquisitionWeighted


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

    acq_list = [(Q, QInt), (IVR_LW, IVR_LWInt), (IVR, IVRInt)]
 
    for (Acq, AcqInt) in acq_list:

        domain = [ [-a, a] for a in 6.0*np.sqrt(np.diag(cov)) ] 
        inputs = GaussianInputs(mean, cov, domain)

        X = inputs.draw_samples(15, "lhs")
        Y = myMap.evaluate(X)
        o = OptimalDesign(X, Y, myMap, inputs, normalize_Y=True)
        x_new = np.atleast_2d([1.0,2.0])
        likelihood = Likelihood(o.model, o.inputs)

        if isinstance(Acq, type(AcquisitionWeighted)):
            acq = Acq(o.model, o.inputs, likelihood=likelihood)
        else:
            acq = Acq(o.model, o.inputs)
        print(acq.evaluate(x_new))
        print(acq.jacobian(x_new))

        domain = [ [-a, a] for a in 20.0*np.sqrt(np.diag(cov)) ] 
        inputs.set_domain(domain)

        if isinstance(Acq, type(AcquisitionWeighted)):
            acqint = AcqInt(o.model, o.inputs, ngrid=250, 
                            likelihood=likelihood)
        else:
            acqint = AcqInt(o.model, o.inputs, ngrid=250)

        print(acqint.evaluate(x_new))
        print(acqint.jacobian(x_new))


if __name__ == "__main__":
    main()
