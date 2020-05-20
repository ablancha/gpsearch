import scipy
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from gpsearch.examples import (AckleyFunction, MichalewiczFunction, BukinFunction)
from gpsearch import custom_KDE, Likelihood, GaussianInputs
import GPy


def plot_likelihood_ratio(function, n_GMM, filename):

    my_map, inputs = function.my_map, function.inputs
   #domain = inputs.domain
   #mean = np.zeros(inputs.input_dim)
   #cov = np.ones(inputs.input_dim)*4
   #inputs = GaussianInputs(domain, mean, cov)

    ngrid = 50
    pts = inputs.draw_samples(n_samples=ngrid, sample_method="grd")
    ndim = pts.shape[-1]
    grd = pts.reshape( (ngrid,)*ndim + (ndim,) ).T
    X, Y = grd[0], grd[1]

    # Compute map
    yy = my_map.evaluate(pts)
    ZZ = yy.reshape( (ngrid,)*ndim ).T
        
    # Compute likelihood ratio
    x, y = custom_KDE(yy, weights=inputs.pdf(pts)).evaluate()
    fnTn = scipy.interpolate.interp1d(x, y)
    fx = inputs.pdf(pts).flatten()
    fy = fnTn(yy).flatten()
    yyg = fx/fy
    ZL = yyg.reshape( (ngrid,)*ndim ).T

    # Compute GMM fit
    kwargs_GMM = dict(n_components=n_GMM, covariance_type="full")
    model = GPy.models.GPRegression(np.random.rand(2,ndim), np.random.rand(2,1))
    likelihood = Likelihood(model, inputs)
    gmm = likelihood._fit_gmm(pts, yyg, kwargs_GMM)
    zg = np.exp(gmm.score_samples(pts))
    ZG = zg.reshape( (ngrid,)*ndim ).T

    # Loop over GMM values
    N = 10
    aic = np.zeros(N)
    bic = np.zeros(N)
    w_raw = yyg - np.min(np.min(yyg))
    from sklearn.mixture import GaussianMixture as GMM
    sca = np.sum(w_raw)
    rng = np.random.default_rng()
    aa = rng.choice(pts, size=1000, p=w_raw/sca)
    for ii in range(1,N):
        kwargs_gmm = dict(n_components=ii, covariance_type="full")
       #gmm = likelihood._fit_gmm(pts, w_raw, kwargs_GMM)
        gmm = GMM(**kwargs_gmm)
        gmm = gmm.fit(X=aa)
        aic[ii] = gmm.aic(pts)
        bic[ii] = gmm.bic(pts)

    fig = plt.figure(figsize=(4.0,1.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.01)

    ax1 = plt.subplot(1, 4, 1)
    plt.contourf(X, Y, ZZ)
    ax2 = plt.subplot(1, 4, 2)
    plt.contourf(X, Y, ZL)
    plt.plot(aa[:,0], aa[:,1], ".", markerfacecolor="pink", markeredgewidth=0, markersize=1)
    ax3 = plt.subplot(1, 4, 3)
    plt.contourf(X, Y, ZG)
    ax4 = plt.subplot(1, 4, 4)
    plt.plot(aic, "-o", lw=0.75, markersize=1)
    plt.plot(bic, "-s", lw=0.75, markersize=1)

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)

    plt.savefig(filename + "_likelihood.pdf")
    plt.close()


if __name__ == "__main__":

    plot_likelihood_ratio(AckleyFunction(), n_GMM=1, 
                          filename="ackley")    

    plot_likelihood_ratio(MichalewiczFunction(), n_GMM=1, 
                          filename="michalewicz")    

