import numpy as np
import time
import GPy
from .utils import set_worker_env
from .minimizers import funmin
from .kernels import *
from .acquisitions.check_acquisition import check_acquisition


class OptimalDesign(object):
    """A class for Bayesian sequential-search algorithm.

    Parameters
    ---------
    X : array
        Array of input points.
    Y : array
        Array of observations.
    my_map: instance of `BlackBox`
        The black-box objective function.
    inputs : instance of `Inputs`
        The input space.
    fix_noise : boolean, optional
        Whether or not to fix the noise variance in the GP model.
    noise_var : float, optional
        Variance for additive Gaussian noise. Default is None, in 
        which case the noise variance from BlackBox is used. If fix_noise 
        is False, noise_var is merely used to initialize the GP model.
    normalize_Y : boolean, optional
        Whether or not to normalize the output in the GP model. 

    Attributes
    ----------
    X, Y, my_map, inputs : see Parameters
    input_dim : int
        Dimensionality of the input space
    model : instance of `GPRegression`
        Current GPy model.

    """

    def __init__(self, X, Y, my_map, inputs, fix_noise=False, 
                 noise_var=None, normalize_Y=True):

        self.my_map = my_map
        self.inputs = inputs
        self.input_dim = inputs.input_dim

        self.X = np.atleast_2d(X)
        self.Y = np.atleast_2d(Y)

        if noise_var is None:
            noise_var = my_map.noise_var

        # Currently only the RBF kernel is supported.
        ker = RBF(input_dim=self.input_dim, ARD=True)

        self.model = GPy.models.GPRegression(X=self.X, 
                                             Y=self.Y, 
                                             kernel=ker, 
                                             normalizer=normalize_Y,
                                             noise_var=noise_var)
        if fix_noise:
            self.model.Gaussian_noise.variance.fix(noise_var)

    def optimize(self, n_iter, acquisition, opt_method="l-bfgs-b", 
                 num_restarts=10, parallel_restarts=False, n_jobs=10, 
                 callback=True, save_iter=True, prefix=None, 
                 kwargs_GPy=None, kwargs_op=None, postpro=False):
        """Runs the Bayesian sequential-search algorithm.

        Parameters
        ----------
        n_iter : int
            Number of iterations (i.e., black-box queries) to perform.
        acquisition : str or instance of `Acquisition`
            Acquisition function for determining the next best point.
            If a string, must be one of 
                - "PI": Probability of Improvement
                - "EI": Expected Improvement
                - "US": Uncertainty Sampling
                - "US-BO": US repurposed for Bayesian Optimization (BO)
                - "US-LW": Likelihood-Weighted US
                - "US-LWBO": US-LW repurposed for BO
                - "US-LWraw" : US-LW with no GMM approximation
                - "US-LWBOraw": US-LWBO with no GMM approximation
                - "LCB" : Lower Confidence Bound
                - "LCB-LW" : Likelihood-Weighted LCB
                - "LCB-LWraw" : LCB-LW with no GMM approximation
                - "IVR" : Integrated Variance Reduction
                - "IVR-IW" : Input-Weighted IVR
                - "IVR-LW" : Likelihood-Weighted IVR
                - "IVR-BO" : IVR repurposed for BO
                - "IVR-LWBO": IVR-LW repurposed for BO
        opt_method : {"L-BFGS-B", "SLSQP", "TNC"}, optional 
            Type of solver. 
        num_restarts : int, optional
            Number of restarts for the optimizer (see funmin).
        parallel_restarts : boolean, optional
            Whether or not to perform optimization in parallel.
        n_jobs : int, optional
            Number of workers used by joblib for parallel computation.
        callback : boolean, optional
            Whether or not to display log at each iteration.
        save_iter : boolean, optional
            Whether or not to save the GP model at each iteration.
        prefix : string, optional
            Prefix for file naming
        kwargs_GPy : dict, optional
            Dictionary of arguments to be passed to the GP model.
        kwargs_op : dict, optional
            Dictionary of arguments to be passed to the scipy optimizer.
        postpro : boolean, optional
            Whether or not to run the sequential-search algorithm in 
            post-processing mode. Loads up GP models saved from 
            previous run.
      
        Returns
        -------
        m_list : list
            A list of trained GP models, one for each iteration of 
            the algorithm.

        """
        if parallel_restarts:
            set_worker_env()

        if prefix is None:
            prefix = "model"

        if kwargs_GPy is None:
            kwargs_GPy = dict(num_restarts=10, optimizer="bfgs", 
                              max_iters=1000, verbose=False) 

        m_list = []
        acq = check_acquisition(acquisition, self.model, self.inputs)

        for ii in range(n_iter+1):

            filename = (prefix+"%.4d")%(ii)
            if postpro:
                if ii == 0 : 
                    print("Running in post-processing mode")
                filename += ".zip"
                model = GPy.models.GPRegression.load_model(filename)
                m_list.append(model.copy())
                self.model = model.copy()
                continue

            if ii == 0:
                self.model.optimize_restarts(**kwargs_GPy)
                m_list.append(self.model.copy())
                if save_iter:
                    self.model.save_model(filename)
                continue

            tic = time.time()

            acq.model = self.model.copy()
            acq.update_parameters()

            xopt = funmin(acq.evaluate, 
                          acq.jacobian, 
                          self.inputs, 
                          opt_method, 
                          kwargs_op=kwargs_op, 
                          num_restarts=num_restarts,
                          parallel_restarts=parallel_restarts, 
                          n_jobs=n_jobs)

            xopt = np.atleast_2d(xopt)
            yopt = self.my_map.evaluate(xopt)

            self.X = np.vstack((self.X, xopt))
            self.Y = np.vstack((self.Y, yopt))
            self.model.set_XY(self.X, self.Y)
            self.model.optimize_restarts(**kwargs_GPy)
            m_list.append(self.model.copy())

            if callback:
                self._callback(ii, time.time()-tic)

            if save_iter:
                self.model.save_model(filename)

        return m_list

    @staticmethod
    def _callback(ii, time):
         m, s = divmod(time, 60)
         print("Iteration {:3d} \t Optimization completed in {:02d}:{:02d}"
               .format(ii, int(m), int(s)))
    
   
