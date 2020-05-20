import scipy
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from gpsearch.examples import (AckleyFunction, MichalewiczFunction, BukinFunction)
from gpsearch import custom_KDE, Likelihood, GaussianInputs, BlackBox
import GPy


def main():

    ndim = 2
    
    domain = [ [-4,4], [-4,4] ]
    mean = np.zeros(ndim)
    cov = np.array([1,1])
    inputs = GaussianInputs(domain, mean, cov)
    my_map = BlackBox(lambda x: 5 + x[0] + x[1] + 2*np.cos(x[0]) + 2*np.sin(x[1]))

   #function = MichalewiczFunction()
   #my_map, inputs = function.my_map, function.inputs
   #domain = inputs.domain
   #mean = np.zeros(inputs.input_dim) + np.pi/2
   #cov = np.ones(inputs.input_dim)*0.1
   #inputs = GaussianInputs(domain, mean, cov)
    
    ngrid = 50
    pts = inputs.draw_samples(n_samples=ngrid, sample_method="grd")
    ndim = pts.shape[-1]
    grd = pts.reshape( (ngrid,)*ndim + (ndim,) ).T
    X, Y = grd[0], grd[1]

    # Compute map
    yy = my_map.evaluate(pts)
    ZZ = yy.reshape( (ngrid,)*ndim ).T

    # Compute input pdf
    pdfx = inputs.pdf(pts)
    PX = pdfx.reshape( (ngrid,)*ndim ).T
        
    # Compute likelihood ratio
    x, y = custom_KDE(yy, weights=inputs.pdf(pts)).evaluate()
    fnTn = scipy.interpolate.interp1d(x, y)
    fx = inputs.pdf(pts).flatten()
    fy = fnTn(yy).flatten()
    yyg = fx/fy
    ZL = yyg.reshape( (ngrid,)*ndim ).T

    # Compute GMM fit
    kwargs_GMM = dict(n_components=2, covariance_type="full")
    model = GPy.models.GPRegression(np.random.rand(2,ndim), np.random.rand(2,1))
    likelihood = Likelihood(model, inputs)
    gmm = likelihood._fit_gmm(pts, yyg, kwargs_GMM)
    zg = np.exp(gmm.score_samples(pts))
    ZG = zg.reshape( (ngrid,)*ndim ).T

    fig = plt.figure(figsize=(3.0,0.9), constrained_layout=True)
    fig = plt.figure(figsize=(6.0,1.65), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.01)

    fs=9
    ax1 = plt.subplot(1, 4, 1)
    plt.contourf(X, Y, ZZ)
    plt.title(r"$f(\mathbf{x})$", fontsize=fs)
    ax2 = plt.subplot(1, 4, 2)
    plt.contourf(X, Y, PX)
    plt.title(r"$p_x(\mathbf{x})$", fontsize=fs)
    ax3 = plt.subplot(1, 4, 3)
    plt.contourf(X, Y, ZL)
    plt.title(r"$w(\mathbf{x})$", fontsize=fs)
    ax4 = plt.subplot(1, 4, 4)
    plt.contourf(X, Y, ZG)
    plt.title(r"$w_\mathit{GMM}(\mathbf{x})$", fontsize=fs)

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)

    plt.savefig("new_likelihood.pdf")
    plt.close()


if __name__ == "__main__":

    main()

