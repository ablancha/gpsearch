import sys
sys.path.append('../../../')

import numpy as np

import numpy as np
from gpsearch import (BlackBox, GaussianInputs,
                      LCB, LCB_LW, IVR, IVR_LW,
                      IVR_LWBO, US_BO, US_LWBO,
                      EI, PI, OptimalDesign, Likelihood,
                      plot_smp, regret_model, distmin_model, comp_pca)

from gpsearch.examples import PRE
#from pre_plotting import plot_observable, plot_trajectory


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
#   print(u[:,-1].max()); exit()
#   plot_observable(t, u[:,2], ylabel="z",
#                   xticks=[0,2000,4000], yticks=[-0.5,0,0.5,1])
#   plot_trajectory(t, u)

    # Do PCA
    iostep = 100
    tsnap = t[::iostep]
    usnap = u[::iostep]
    lam, psi, usnap_mean = comp_pca(usnap, ndim)
    print("PCA Eigenvalues:", lam)

    n_init = 3
    n_core = 0
    n_iter = 50#140

    mean = np.zeros(ndim)
    cov = np.diag(lam)
    a_bnds = 4.0
    domain = [ [-a, a] for a in a_bnds*np.sqrt(np.diag(cov)) ]
    inputs = GaussianInputs(domain, mean, cov)

    my_map = BlackBox(map_def, args=(psi,usnap_mean))

    X = inputs.draw_samples(n_init, "lhs")
    Y = my_map.evaluate(X, parallel=True)

    o = OptimalDesign(X, Y, my_map, inputs,
                      fix_noise=True,
                      noise_var=0.0,
                      normalize_Y=True)

    acquisition = EI(o.model, o.inputs)
   #acquisition = IVR_LWBO(o.model, o.inputs)
    m_list = o.optimize(n_iter,
                        acquisition=acquisition,
                        num_restarts=10,
                        parallel_restarts=True)#, postpro=True)

if __name__ == "__main__":
    main()


