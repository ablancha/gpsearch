import numpy as np
from gpsearch import (BlackBox, GaussianInputs, OptimalDesign, 
                      Benchmarker, BenchmarkerLHS, comp_pca)
from gpsearch.examples import PRE


def map_def(theta, psi, usnap_mean):
    tf = 50
    dt = 0.01
    nsteps = int(tf/dt)
    u_init = np.dot(psi, theta) + usnap_mean
    pre = PRE(tf, nsteps, u_init)
    u, t = pre.solve()
    z = u[:,-1]
    return -np.max(z)


def main():

    # Generate data for PCA
    tf = 2000
    dt = 0.01
    nsteps = int(tf/dt)
    u_init = [0, 0.01, 0.01]
    pre = PRE(tf, nsteps, u_init)
    u, t = pre.solve()

    # Do PCA
    ndim = 2
    iostep = 100
    tsnap = t[::iostep]
    usnap = u[::iostep]
    lam, psi, usnap_mean = comp_pca(usnap, ndim)
    print("PCA Eigenvalues:", lam)

    # Set up black box
    mean = np.zeros(ndim)
    cov = np.diag(lam)
    a_bnds = 4.0
    domain = [ [-a, a] for a in a_bnds*np.sqrt(np.diag(cov)) ]
    inputs = GaussianInputs(domain, mean, cov)

    noise_var = 1e-3 
    my_map = BlackBox(map_def, args=(psi,usnap_mean), noise_var=noise_var)

    # Do Benchmark
    prefix = "prec2d_"

    n_init = 3
    n_iter = 100
    n_trials = 100
    n_jobs = 40

    metric = [ ("regret_tmap", dict(tmap=my_map,true_ymin=0.0)),
               ("regret_obs", dict(true_ymin=0.0)) ]

    acq_list = ["EI", "PI", "LCB", "LCB_LW",
                "IVR", "IVR_BO", "IVR_LW", "IVR_LWBO"]

    for acq in acq_list:
        print("Benchmarking " + acq)
        b = Benchmarker(my_map, acq, n_init, n_iter, inputs, metric)
        result = b.run_benchmark(n_trials, n_jobs, filename=prefix+acq)


if __name__ == "__main__":
    main()


