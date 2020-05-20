import numpy as np
from gpsearch import (OptimalDesign, Benchmarker)
from gpsearch.examples import AckleyFunction


def main():

    noise_var = 1e-3 

    b = AckleyFunction(noise_var=noise_var, rescale_X=True)
    my_map, inputs, true_ymin, true_xmin = b.my_map, b.inputs, b.ymin, b.xmin

    prefix = "ackley2d_noisevar1e-3_"

    n_init = 3
    n_iter = 70
    n_trials = 100
    n_jobs = 40

    metric = [ ("regret_tmap", dict(true_ymin=true_ymin,tmap=my_map)),
               ("regret_obs", dict(true_ymin=true_ymin)) ]

    acq_list = ["EI", "IVR_LWBO"]

    for acq in acq_list:
        print("Benchmarking " + acq)
        b = Benchmarker(my_map, acq, n_init, n_iter, inputs, metric)
        result = b.run_benchmark(n_trials, n_jobs, filename=prefix+acq)


if __name__ == "__main__":
    main()


