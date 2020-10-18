import numpy as np
from gpsearch import (BenchmarkerPath, PathPlanner, Ackley, Bird,
                      RosenbrockModified, Michalewicz, Bukin, 
                      custom_KDE, GaussianInputs)


def run(function, prefix):

    noise_var = 1e-3

    b = function(noise_var=noise_var, rescale_X=True)
    my_map, inputs, true_ymin, true_xmin = b.my_map, b.inputs, b.ymin, b.xmin

#   domain = inputs.domain
#   mean = np.zeros(inputs.input_dim) + 0.5
#   cov = np.ones(inputs.input_dim)*0.01
#   inputs = GaussianInputs(domain, mean, cov)

    record_time = np.linspace(0,7,101)
    n_trials = 100
    n_jobs = 40

    pts = inputs.draw_samples(n_samples=int(1e5), sample_method="uni")
    yy = my_map.evaluate(pts, parallel=True, include_noise=False)
    pt = custom_KDE(yy, weights=inputs.pdf(pts))

    X_pose = (0,0, np.pi/4)
    planner = PathPlanner(inputs.domain, 
                          look_ahead=0.2, 
                          turning_radius=0.02,
                          n_frontier=150,
                          fov=0.75*np.pi)

    metric = [ ("rmse",          dict(pts=pts, yy=yy)) ,
               ("log_pdf",       dict(pt=pt, pts=pts)) ,
               ("distmin_model", dict(true_xmin=true_xmin)) ,
               ("regret_tmap"  , dict(true_ymin=true_ymin, tmap=my_map)) ,
               ("regret_obs"   , dict(true_ymin=true_ymin)) ]

    acq_list = ["US", "US_LW", "IVR", "IVR_LW"]

    for acq in acq_list:
        print("Benchmarking " + acq)
        b = BenchmarkerPath(my_map, acq, planner, X_pose, record_time, inputs, metric=metric)
        result = b.run_benchmark(n_trials, n_jobs=n_jobs, filename=prefix+acq)


if __name__ == "__main__":
    run(Ackley, "ackley_"); 
    run(Bird, "bird_")
    run(Bukin, "bukin_")
    run(Michalewicz, "micha_")
    run(RosenbrockModified, "rosenmod_")

