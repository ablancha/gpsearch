import numpy as np
from gpsearch import (BlackBox, GaussianInputs, OptimalDesign, comp_pca)
from pre_plotting import plot_observable, plot_trajectory
from precursor import PRE


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

    ndim = 3
    np.random.seed(2)

    tf = 4000
    dt = 0.01
    nsteps = int(tf/dt)
    u_init = [0, 0.01, 0.01]
    pre = PRE(tf, nsteps, u_init)
    u, t = pre.solve()
    plot_observable(t, u[:,2], ylabel="z",
                    xticks=[0,2000,4000], yticks=[-0.5,0,0.5,1])
    plot_trajectory(t, u)

    # Do PCA
    iostep = 100
    tsnap = t[::iostep]
    usnap = u[::iostep]
    lam, psi, usnap_mean = comp_pca(usnap, ndim)
    print("PCA Eigenvalues:", lam)

    n_init = 3
    n_iter = 50

    mean = np.zeros(ndim)
    cov = np.diag(lam)
    a_bnds = 4.0
    domain = [ [-a, a] for a in a_bnds*np.sqrt(np.diag(cov)) ]
    inputs = GaussianInputs(domain, mean, cov)
    my_map = BlackBox(map_def, args=(psi,usnap_mean))

    X = inputs.draw_samples(n_init, "lhs")
    Y = my_map.evaluate(X)

    o = OptimalDesign(X, Y, my_map, inputs,
                      fix_noise=True,
                      noise_var=0.0,
                      normalize_Y=True)

    m_list = o.optimize(n_iter,
                        acquisition="EI",
                        num_restarts=10,
                        parallel_restarts=True)


if __name__ == "__main__":
    main()


