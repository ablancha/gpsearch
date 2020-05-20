import numpy as np
from gpsearch import (BlackBox, GaussianInputs, OptimalDesign,\
                      Benchmarker, BenchmarkerLHS,\
                      US, US_LW, IVR, IVR_LW, true_pdf, error_logpdf)
from gpsearch.examples import Oscillator, Noise


def map_def(theta, oscil):
    u, t = oscil.solve(theta)
    mean_disp = np.mean(u[:,0])
    return mean_disp


def main():

    ndim = 2
    tf = 25
    nsteps = 1000
    u_init = [0, 0]
    noise = Noise([0, tf])
    oscil = Oscillator(noise, tf, nsteps, u_init)
    myMap = BlackBox(map_def, args=(oscil,))

    mean, cov = np.zeros(ndim), np.ones(ndim)
    domain = [ [-6, 6] ] * ndim
    inputs = GaussianInputs(domain, mean, cov)
    pt = true_pdf(inputs, "map_samples2D.txt")

    prefix = "oscill_" 

    n_init = 4
    n_iter = 1#80
    n_trials = 10#200
    n_jobs = 10#20

    metric = [ ("log_pdf", dict(pt=pt)) ]
    acq_list = ["IVR", "US_LW", "IVR", "IVR_LW"] 
    
    for acq in acq_list:
        b = Benchmarker(myMap, acq, n_init, n_iter, inputs, metric)
        result = b.run_benchmark(n_trials, n_jobs, filename=prefix+acq)

    b = BenchmarkerLHS(myMap, n_init, n_iter, inputs, metric)
    result = b.run_benchmark(n_trials, n_jobs, filename=prefix+"LHS")


if __name__ == "__main__":
    main()


