import sys
sys.path.append('../../')

import numpy as np
import time
from scipy import stats
from matplotlib import pyplot as plt
from gpsearch import GaussianInputs, KDE_Numba
from gpsearch.examples import Oscillator, Noise
from KDEpy import FFTKDE
import statsmodels.api as sm


def benchmark_gumbel(n_run=1):

    for ii in range(1,5):

        toc_scipy = 0.0
        toc_numba = 0.0
        toc_kdepy = 0.0
        toc_stats = 0.0

        for nn in range(n_run):

            mu, beta = 0, 0.1 
            smpl = np.random.gumbel(mu, beta, int(10**(ii)))
            x_d = np.linspace(np.min(smpl)-0.01*np.abs(np.min(smpl)),
                              np.max(smpl)+0.01*np.abs(np.max(smpl)), 10000)
            weights = np.ones(smpl.shape)
            bw = KDE_Numba(smpl, weights=weights).bw

            tic = time.time()
            pdf_scipy = stats.gaussian_kde(smpl, weights=weights)(x_d)
            toc_scipy += time.time() - tic

            tic = time.time()
            pdf_numba = KDE_Numba(smpl, weights=weights)(x_d)
            toc_numba += time.time() - tic

            tic = time.time()
            pdf_kdepy = FFTKDE(bw=bw).fit(smpl, weights)(x_d)
            toc_kdepy += time.time() - tic

            tic = time.time()
            dens = sm.nonparametric.KDEUnivariate(smpl)
            dens.fit(bw=bw, weights=weights, fft=False)
            pdf_stats = dens.evaluate(x_d)
            toc_stats += time.time() - tic
 
        print(ii, toc_scipy/n_run, toc_numba/n_run, toc_kdepy/n_run, toc_stats/n_run)


def compare_pdf_Oscillator():

    smpl = np.genfromtxt("map_samples2D.txt")
    ndim = 2
    tf = 25
    nsteps = 1000
    u_init = [0, 0]
    noise = Noise([0, tf])
    lam = noise.get_eigenvalues(ndim)
    mean = np.zeros(ndim)
    cov = np.diag(lam)
    domain = [ [-a, a] for a in 6.0*np.sqrt(np.diag(cov)) ]
    inputs = GaussianInputs(mean, cov, domain)
    weights = inputs.pdf(smpl[:,0:-1])
    x_d = np.linspace(-3,3,500) 

   #weights = weights/weights
    pdf_scipy = stats.gaussian_kde(smpl[:,-1], weights=weights)
    pdf_numba = KDE_Numba(smpl[:,-1], weights=weights)

    pdf_kdepy = FFTKDE(bw=pdf_numba.bw).fit(smpl[:,-1], weights)

    plt.semilogy(x_d, pdf_scipy(x_d), lw=3)
    plt.semilogy(x_d, pdf_numba(x_d), '--')
    plt.semilogy(x_d, pdf_kdepy(x_d), '--', lw=0.5)
    plt.xlim(-3, 3)
    plt.ylim(1e-8, 1e2)
    plt.show()


if __name__ == "__main__":
   #benchmark_gumbel(20)
    compare_pdf_Oscillator()

