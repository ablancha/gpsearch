import numpy as np
from gpsearch import Benchmarker
from gpsearch.examples import Ackley


def main():

    prefix = "ackley_"
    noise_var = 1e-3 
    b = Ackley(noise_var=noise_var, rescale_X=True)
    my_map, inputs, true_ymin, true_xmin = b.my_map, b.inputs, b.ymin, b.xmin

    n_init = 3
    n_iter = 80
    n_trials = 100
    n_jobs = 40

    metric = [ ("distmin_model", dict(true_xmin=true_xmin)) ,
               ("distmin_obs"  , dict(true_xmin=true_xmin)) ,
               ("regret_tmap"  , dict(true_ymin=true_ymin, tmap=my_map)) ,
               ("regret_model" , dict(true_ymin=true_ymin)) ,
               ("regret_obs"   , dict(true_ymin=true_ymin)) ]

    acq_list = ["EI", "PI", "LCB", "LCB_LW",
                "IVR", "IVR_BO", "IVR_LW", "IVR_LWBO"]

    for acq in acq_list:
        print("Benchmarking " + acq)
        b = Benchmarker(my_map, acq, n_init, n_iter, inputs, metric)
        result = b.run_benchmark(n_trials, n_jobs, filename=prefix+acq)


if __name__ == "__main__":
    main()


