import numpy as np
from gpsearch import (BlackBox, GaussianInputs, OptimalDesign, 
                      true_pdf, model_pdf, plot_pdf, plot_smp)
from gpsearch.core.acquisitions import *
from oscillator import Oscillator, Noise


def map_def(theta, oscil):
    u, t = oscil.solve(theta)
    mean_disp = np.mean(u[:,0])
    return mean_disp


def main():

    ndim = 2
    np.random.seed(3)

    tf = 25
    nsteps = 1000
    u_init = [0, 0]
    noise = Noise([0, tf])
    oscil = Oscillator(noise, tf, nsteps, u_init)
    my_map = BlackBox(map_def, args=(oscil,))

    n_init = 4
    n_iter = 80

    mean, cov = np.zeros(ndim), np.ones(ndim)
    domain = [ [-6, 6] ] * ndim
    inputs = GaussianInputs(domain, mean, cov)
    X = inputs.draw_samples(n_init, "lhs")
    Y = my_map.evaluate(X)

    o = OptimalDesign(X, Y, my_map, inputs, 
                      fix_noise=True, 
                      noise_var=0.0, 
                      normalize_Y=True)
    m_list = o.optimize(n_iter,
                        acquisition="IVR_LW",
                        num_restarts=10,
                        parallel_restarts=True)

    fname_map = "map_samples2D.txt"
    pt = true_pdf(inputs, filename=fname_map)
    pts = inputs.draw_samples(n_samples=100, sample_method="grd")

    for ii in np.arange(0, n_iter+1, 10):
        plot_smp(m_list[ii], inputs, n_init,
                 filename="smps%.4d.pdf"%(ii), 
                 xticks=[-6,0,6], yticks=[-5,0,5], 
                 cmapticks=[-2,-1,0,1,2])


if __name__ == "__main__":
    main()


