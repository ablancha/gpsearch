import scipy
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from gpsearch.examples import (AckleyFunction, MichalewiczFunction, BukinFunction)
from gpsearch import custom_KDE, Likelihood, GaussianInputs
import GPy
from GPy.models import GradientChecker


def plot_likelihood_ratio(function, n_GMM, filename):

    my_map, inputs = function.my_map, function.inputs
    mu = np.random.randn(inputs.input_dim)
    cov = 4*np.random.randn(inputs.input_dim)**2
    inputs = GaussianInputs(inputs.domain, mu=mu, cov=cov)

    ngrid = 10
    pts = inputs.draw_samples(n_samples=ngrid, sample_method="grd")
    ndim = pts.shape[-1]
    grd = pts.reshape( (ngrid,)*ndim + (ndim,) ).T
    X, Y = grd[0], grd[1]

    # Compute map
    yy = my_map.evaluate(pts)
    # Compute GPy model
    model = GPy.models.GPRegression(pts, yy, normalizer=True)

    likelihood = Likelihood(model, inputs)
    x_new = np.random.rand(2, inputs.input_dim) 
    print(likelihood._evaluate_raw(x_new))
    print(likelihood._jacobian_raw(x_new))

    g = GradientChecker(lambda x: likelihood._evaluate_gmm(x),
                        lambda x: likelihood._jacobian_gmm(x),
                        x_new, 'x')
    assert(g.checkgrad())


    yy = model.predict(pts)[0].flatten()
    dyy_dx, _ = model.predictive_gradients(pts)
    dyy_dx = dyy_dx[:,:,0]

    ZZ = yy.reshape( (ngrid,)*ndim ).T
        
    # Compute likelihood ratio
    x, y = custom_KDE(yy, weights=inputs.pdf(pts)).evaluate()
    fnTn = scipy.interpolate.interp1d(x, y)
    fx = inputs.pdf(pts).flatten()
    fy = fnTn(yy).flatten()
    w = fx/fy
    ZL = w.reshape( (ngrid,)*ndim ).T

    # Compute gradient of likelihood ratio
    dy_dx = np.gradient(y,x)
    fnTn_dx = scipy.interpolate.interp1d(x, dy_dx)
    tmp = -fx / fy**2 * fnTn_dx(yy) 
    dw_dx = tmp[:,None] * dyy_dx

    plt.figure()
    plt.plot(x,y)
    plt.plot(x,fnTn(x), '-.')
    from scipy.interpolate import InterpolatedUnivariateSpline
    spl = InterpolatedUnivariateSpline(x, y)
    plt.plot(x, spl(x), '--')

    plt.figure()
    plt.semilogy(x,dy_dx)
    plt.semilogy(x,fnTn_dx(x), '-.')
    plt.semilogy(x, spl.derivative()(x), '--')
    plt.show(); exit()


if __name__ == "__main__":

    plot_likelihood_ratio(AckleyFunction(rescale_X=True), n_GMM=2, 
                          filename="ackley")    

    plot_likelihood_ratio(MichalewiczFunction(), n_GMM=4, 
                          filename="michalewicz")    

