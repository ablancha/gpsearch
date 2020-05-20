import numpy as np
from joblib import Parallel, delayed
from ..core.optimaldesign import OptimalDesign
from ..core.optimalpath import OptimalPath
from ..core.utils import set_worker_env
from ..core.metrics import *


class Benchmarker(object):
    """A class for benchmark of sequential-search algorithm.

    Parameters
    ----------
    tmap : instance of `BlackBox`
        The black box.
    acquisition : str or instance of `Acquisition`
        The acquisition function (see OptimalDesign.optimize doc).
    n_init : int
        Number of initial points. Drawn from an LHS design.
    n_iter : int
        Number of iterations (i.e., black-box queries) to perform.
    inputs : instance of `Inputs`
        The input space.
    metric : list of tuples
        A list of error metrics used for benchmarking. Must be of the 
        form [ (metric_1, kwargs_metric_1), ... ], where `metric_1` is 
        one of
            - "rmse": RMSE between true and estimated objective function
            - "log_pdf": log-distance between true and estimated pdf
            - "regret_tmap": instantaneous regret using the true map
            - "regret_model": instantaneous regret using GP model
            - "regret_obs": instantaneous regret using past observations
            - "distmin_model": distance to minimum using GP model
            - "distmin_obs": distance to minimum using past observations
        and `kwargs_metric_1` is a dictionary of keyword arguments for
        evaluation of `metric_1`.

    Attributes
    ----------
    tmap, acquisition, n_init, n_iter, inputs, metric : see Parameters

    """

    def __init__(self, tmap, acquisition, n_init, n_iter, inputs, 
                 metric):
        self.tmap = tmap
        self.acquisition = acquisition
        self.n_init = n_init
        self.n_iter = n_iter
        self.inputs = inputs
        self.metric = [ (self.set_metric(met), kwa) 
                        for (met, kwa) in metric ]

    def run_benchmark(self, n_trials, n_jobs=20, save_res=True,
                      filename=None):
        """Benchmark of sequential-search algorithm.

        Parameters
        ---------- 
        n_trials : int
            Number of simulations to run. Each simulation corresponds 
            to a different (random) choice of initial points.
        n_jobs : int, optional
            Number of workers used by joblib for parallel computation.
        save_res : boolean, optional
            Whether or not to save the benchmark results to disk.
        filename : str, optional
            Filename for saving results.
            
        """
        res = self._run_benchmark(n_trials, n_jobs=n_jobs)
        if save_res:
            self._save_benchmark(res, filename)
        return res

    def _run_benchmark(self, n_trials, n_jobs=20):
        if n_jobs > 1:
            set_worker_env()
        x = Parallel(n_jobs=n_jobs, backend="loky", verbose=10) \
                    ( delayed(self.optimization_loop)(self.tmap,
                                                      self.acquisition, 
                                                      self.n_init, 
                                                      self.n_iter,
                                                      self.inputs,
                                                      self.metric,
                                                      ii) \
                      for ii in range(n_trials) )
        return np.array(x)

    def _save_benchmark(self, res, filename):
        if filename is None:
            filename = "bench_res"
        for ii, (met, kwa) in enumerate(self.metric):
            met_name = met.__name__
            np.save(filename + "_" + met_name, res[:,ii,:])

    @staticmethod
    def optimization_loop(tmap, acquisition, n_init, n_iter, inputs, 
                          metric, seed=None):
        np.random.seed(seed)
        X = inputs.draw_samples(n_init, "lhs")
        Y = tmap.evaluate(X)
        o = OptimalDesign(X, Y, tmap, inputs,
                          fix_noise=False,
                          noise_var=None,
                          normalize_Y=True)
        m_list = o.optimize(n_iter, acquisition,
                            callback=False, 
                            save_iter=False,
                            num_restarts=10,
                            parallel_restarts=False)
        result = [ met(m_list, inputs, **kwa) for (met, kwa) in metric ]
        return result

    @staticmethod
    def set_metric(metric):
        if isinstance(metric, str):
            if metric.lower() == "rmse":
                return rmse
            elif metric.lower() == "log_pdf":
                return log_pdf
            elif metric.lower() == "regret_tmap":
                return regret_tmap
            elif metric.lower() == "regret_model":
                return regret_model
            elif metric.lower() == "regret_obs":
                return regret_obs
            elif metric.lower() == "distmin_model":
                return distmin_model
            elif metric.lower() == "distmin_obs":
                return distmin_obs
            else:
                raise NotImplementedError
        elif callable(metric):
            return metric
        else:
            raise ValueError


class BenchmarkerLHS(Benchmarker):
    """A class for benchmark of LHS sampling.

    Parameters
    ----------
    tmap, inputs, metric : see parent class (`Benchmarker')
    n_init : int
        Number of initial points. Drawn from an LHS design.
    n_iter : int
        Maximum number of LHS points.

    Attributes
    ----------
    tmap, inputs, metric : see Parameters
    n_total : int
        Same as `n_iter`.
    n_sampl : int
        Same as `n_init`.

    Notes
    -----
    Since LHS sampling is a non-iterative algorithm, the parent class
    `Benchmarker` may not be used as is. This subclass repurposes the 
    parent class by overriding the `run_benchmark` method and redefining
    the `n_iter` and `n_init` parameters. 

    """

    def __init__(self, tmap, n_init, n_iter, inputs, metric):
        super().__init__(tmap, "US", n_init, 0, inputs, metric)
        self.n_total = n_iter
        self.n_sampl = n_init

    def run_benchmark(self, n_trials, n_jobs=20, save_res=True,
                      filename=None):
        """Benchmark of sequential-search algorithm.

        Parameters
        ---------- 
        n_trials : int
            Number of simulations to run. Each simulation corresponds 
            to a different (random) choice of initial points.
        n_jobs : int, optional
            Number of workers used by joblib for parallel computation.
        save_res : boolean, optional
            Whether or not to save the benchmark results to disk.
        filename : str, optional
            Filename for saving results.
            
        """
        res = np.empty(shape=[n_trials, len(self.metric), 0])

        for ii in range(self.n_total):
            print("Iteration {:3d}".format(ii))
            self.n_init = self.n_sampl + ii
            tmp = self._run_benchmark(n_trials, n_jobs=n_jobs)
            res = np.dstack((res, tmp))

        if save_res:
            self._save_benchmark(res, filename)

        return res


class BenchmarkerPath(Benchmarker):

    def __init__(self, tmap, acquisition, path_planner, X_pose,
                 n_iter, inputs, metric):
        super().__init__(tmap, acquisition, 0, n_iter, inputs,
                         metric)
        self.path_planner = path_planner
        self.X_pose = X_pose

    def _run_benchmark(self, n_trials, n_jobs=20):
        if n_jobs > 1:
            set_worker_env()
        x = Parallel(n_jobs=n_jobs, backend="loky", verbose=10) \
                    ( delayed(self.optimization_loop)(self.tmap,
                                                      self.acquisition,
                                                      self.path_planner,
                                                      self.X_pose,
                                                      self.n_iter,
                                                      self.inputs,
                                                      self.metric,
                                                      ii) \
                      for ii in range(n_trials) )
        return np.array(x)

    @staticmethod
    def optimization_loop(tmap, acquisition, path_planner, X_pose,
                          n_iter, inputs, metric, seed=None):
        np.random.seed(seed)
        Y_pose = tmap.evaluate(X_pose[0:2])
        o = OptimalPath(X_pose, Y_pose, tmap, inputs,
                        fix_noise=False,
                        noise_var=None,
                        normalize_Y=True)
        m_list, _ = o.optimize(n_iter, acquisition, path_planner,
                               callback=False,
                               save_iter=False)
        result = [ met(m_list, inputs, **kwa) for (met, kwa) in metric ]
        return result

