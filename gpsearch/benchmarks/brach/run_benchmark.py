import numpy as np
from scipy import interpolate
from brachistochrone import travel_time, cycloid
from gpsearch import (BlackBox, UniformInputs, OptimalDesign, Benchmarker, 
                      GaussianInputs)


def map_def(theta, x, y2, tc):
    y = np.hstack(([0], theta, [y2]))
    return np.log10(travel_time(x, y)-tc)


def main():

    ndim = 10

    x2, y2 = 1, 0.3
    x = np.linspace(0, x2, ndim+2)
    x_pts = x[1:-1]

    xc, yc = cycloid(x2, y2, N=200)
    tc = travel_time(xc, yc) # Best ever
    ff = interpolate.interp1d(xc,yc)

    noise_var = 1e-2
    my_map = BlackBox(map_def, args=(x,y2,tc), noise_var=noise_var)
    domain = [ [0.05, 2*y2] for n in np.arange(0,ndim) ]
    inputs = UniformInputs(domain)

    prefix = "tclog_noise2_"

    n_init = 10
    n_iter = 300
    n_trials = 40
    n_jobs = 40

    true_ymin = 0.0
    metric = [ ("regret_tmap"  , dict(true_ymin=true_ymin, tmap=my_map)) ,
               ("regret_obs"   , dict(true_ymin=true_ymin)) ]

    acq_list = ["EI", "PI", "LCB", "LCB_LW", "LCB_LWraw", "IVR_BO", "IVR_LWBO"]

    for acq in acq_list:
        print("Benchmarking " + acq)
        b = Benchmarker(my_map, acq, n_init, n_iter, inputs, metric=metric)
        result = b.run_benchmark(n_trials, n_jobs=n_jobs, filename=prefix+acq)


if __name__ == "__main__":
    main()


