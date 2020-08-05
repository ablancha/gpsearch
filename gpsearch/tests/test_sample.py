import numpy as np
import GPy
from GPy.models import GradientChecker


def test_predictive_gradients_with_normalizer():
    """
    Check that model.predictive_gradients returns the gradients of
    model.predict when normalizer=True 
    """
    N, M, Q = 10, 15, 3
    X = np.random.rand(M,Q)
    Y = np.random.rand(M,1)
    x = np.random.rand(N,Q)
    model = GPy.models.GPRegression(X=X, Y=Y, normalizer=False)
    gm = GradientChecker(lambda x: model.predict(x)[0],
                         lambda x: model.predictive_gradients(x)[0],
                         x, 'x')
    gc = GradientChecker(lambda x: model.predict(x)[1],
                         lambda x: model.predictive_gradients(x)[1],
                         x, 'x')
    assert(gm.checkgrad())
    assert(gc.checkgrad())


def test_posterior_covariance_between_points_with_normalizer():
    """
    Check that model.posterior_covariance_between_points returns 
    the covariance from model.predict when normalizer=True
    """
    N, M, Q = 10, 15, 3
    X = np.random.rand(M,Q)
    Y = np.random.rand(M,1)
    x = np.random.rand(N,Q)
    model = GPy.models.GPRegression(X=X, Y=Y, normalizer=False)

    c1 = model.posterior_covariance_between_points(x,x)
    c2 = model.predict(x, full_cov=True)[1]
    np.testing.assert_allclose(c1,c2)






