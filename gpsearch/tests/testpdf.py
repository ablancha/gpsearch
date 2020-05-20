import sys
sys.path.append('../../')

import numpy as np
import gpsearch
import GPy
from GPy.models import GradientChecker
from gpsearch.core import jacobian_fdiff, GaussianInputs, \
                          LogNormalInputs, UniformInputs, \
                          Likelihood, US, US_LW, LCB, LCB_LW, \
                          EI, PI, IVR, IVR_LW, US_BO, US_LWBO
from gpsearch.core.kernels import RBF


def main():

    np.random.seed(2)

    M, Q = 15, 5
    X = np.random.rand(M,Q)
    Y = np.random.rand(M,1)

    mu = np.random.rand(Q)
    cov = np.random.rand(Q)**2

    lb = np.abs(np.random.randn(Q,1))
    ub = lb + np.abs(np.random.randn(Q,1))
    domain = np.hstack((lb,ub)).tolist()
    inputs = UniformInputs(domain)
   #inputs = GaussianInputs(domain, mu, cov)
   #inputs = LogNormalInputs(domain, mu, cov)

    x_new = np.random.rand(3, Q)

    g = GradientChecker(lambda x: inputs.pdf(x),
                        lambda x: inputs.pdf_jac(x),
                        x_new, 'x')
    assert(g.checkgrad())


if __name__ == "__main__":
    main()
