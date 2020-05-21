import numpy as np
from gpsearch.examples import Oscillator, Noise
from gpsearch import (BlackBox, GaussianInputs, OptimalDesign,
                      Benchmarker, BenchmarkerLHS, custom_KDE)


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

    mean, cov = np.zeros(ndim), np.ones(ndim)
    domain = [ [-6, 6] ] * ndim
    inputs = GaussianInputs(domain, mean, cov)

    prefix = "oscill_" 
    noise_var = 1e-3 
    my_map = BlackBox(map_def, args=(oscil,), noise_var=noise_var)

    n_init = 3
    n_iter = 80
    n_trials = 100
    n_jobs = 30

    # Compute true pdf
    filename = "map_samples{:d}D.txt".format(ndim)
    try:
        smpl = np.genfromtxt(filename)
        pts = smpl[:,0:-1]
        yy = smpl[:,-1]
    except:
        pts = inputs.draw_samples(n_samples=100, sample_method="grd")
        yy = my_map.evaluate(pts, parallel=True, include_noise=False)
        np.savetxt(filename, np.column_stack((pts,yy)))
    weights = inputs.pdf(pts)
    pt = custom_KDE(yy, weights=weights)

    metric = [ ("log_pdf", dict(pt=pt, pts=pts)) ]
    acq_list = ["US", "US_LW", "IVR_IW", "IVR_LW"]

    for acq in acq_list:
        print("Benchmarking " + acq)
        b = Benchmarker(my_map, acq, n_init, n_iter, inputs, metric)
        result = b.run_benchmark(n_trials, n_jobs, filename=prefix+acq)

    b = BenchmarkerLHS(my_map, n_init, n_iter, inputs, metric)
    result = b.run_benchmark(n_trials, n_jobs, filename=prefix+"LHS")


if __name__ == "__main__":
    main()


