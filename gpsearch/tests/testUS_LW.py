import sys
sys.path.append('../../')

import numpy as np
import gpsearch
import GPy
from GPy.models import GradientChecker
from gpsearch.core import jacobian_fdiff, UniformInputs, \
                          Likelihood, US, US_LW, LCB, LCB_LW, \
                          EI, PI, IVR, IVR_LW, US_BO, US_LWBO
from gpsearch.core.kernels import RBF


def main():

    np.random.seed(2)

    M, Q = 15, 3
    X = np.random.rand(M,Q)
    Y = np.random.rand(M,1)

    ker = RBF(input_dim=Q, ARD=True, variance=1.34, 
              lengthscale=np.random.rand(1,Q))
    model = GPy.models.GPRegression(X=X, Y=Y, kernel=ker,
                                    normalizer=True)

    inputs = UniformInputs([[0,1]]*Q)
    likelihood = Likelihood(model, inputs)

    x_new = np.random.rand(2, Q)
    qcrit = LCB_LW(model, inputs, likelihood=likelihood)

    g = GradientChecker(lambda x: qcrit.evaluate(x),
                        lambda x: qcrit.jacobian(x),
                        x_new, 'x')
    assert(g.checkgrad())

    a = qcrit.evaluate(x_new)
    b = a+0.0
    for i in range(x_new.shape[0]):
        print(qcrit.evaluate(x_new[i,:]))
    print(a)

    a = qcrit.jacobian(x_new)
    b = a+0.0
    for i in range(x_new.shape[0]):
        print(qcrit.jacobian(x_new[i,:]))
    print(a)


if __name__ == "__main__":
    main()
