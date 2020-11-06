import numpy as np
import time
import GPy
from .kernels import *
from .acquisitions.check_acquisition import check_acquisition


class OptimalPath(object):
    """A class for Bayesian path-planning algorithm.

    Parameters
    ---------
    X_pose : array_like
        The robot's starting pose. Must be in the form 
        (x_pos, y_pos, angle)
    Y_pose : array_like
        Observation at starting position.
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
    my_map, inputs : see Parameters
    X : array
        Array of input points.
    Y : array
        Array of observations.
    X_pose : array_like
        The robot's current pose. 
    input_dim : int
        Dimensionality of the input space
    model : instance of `GPRegression`
        Current GPy model.

    """

    def __init__(self, X_pose, Y_pose, my_map, inputs, fix_noise=False, 
                 noise_var=None, normalize_Y=True):

        self.my_map = my_map
        self.inputs = inputs
        self.input_dim = inputs.input_dim

        self.X = np.atleast_2d(X_pose[0:2])
        self.Y = np.atleast_2d(Y_pose)
        self.X_pose = X_pose

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

    def optimize(self, record_time, acquisition, path_planner, 
                 callback=True, save_iter=True, prefix=None, kwargs_GPy=None):
                 
        """Runs the Bayesian path-planning algorithm.

        Parameters
        ----------
        record_time : array_like
            Time vector for when measurements are made.
        acquisition : str or instance of `Acquisition`
            Acquisition function for determining the next best point.
            If a string, must be one of 
                - "PI": Probability of Improvement
                - "EI": Expected Improvement
                - "US": Uncertainty Sampling
                - "US_BO": US repurposed for Bayesian Optimization (BO)
                - "US_LW": Likelihood-Weighted US
                - "US_LWBO": US-LW repurposed for BO
                - "US_LWraw" : US-LW with no GMM approximation
                - "US_LWBOraw": US-LWBO with no GMM approximation
                - "LCB" : Lower Confidence Bound
                - "LCB_LW" : Likelihood-Weighted LCB
                - "LCB_LWraw" : LCB-LW with no GMM approximation
                - "IVR" : Integrated Variance Reduction
                - "IVR_IW" : Input-Weighted IVR
                - "IVR_LW" : Likelihood-Weighted IVR
                - "IVR_BO" : IVR repurposed for BO
                - "IVR_LWBO": IVR-LW repurposed for BO
        path_planner : instance of `PathPlanner`
            Parametrization of the path by Dubins Curves.
        callback : boolean, optional
            Whether or not to display log at each iteration.
        save_iter : boolean, optional
            Whether or not to save the GP model at each iteration.
        prefix : string, optional
            Prefix for file naming
        kwargs_GPy : dict, optional
            Dictionary of arguments to be passed to the GP model.
      
        Returns
        -------
        m_list : list
            A list of trained GP models, one for each iteration of 
            the algorithm.
        p_list : list
            A list of paths, one for each iteration of the algorithm. 

        """
        if prefix is None:
            prefix = "model"

        if kwargs_GPy is None:
            kwargs_GPy = dict(num_restarts=10, optimizer="bfgs", 
                              max_iters=1000, verbose=False) 

        m_list, p_list = [], []

        t_cur = 0 
        t_fin = record_time[-1]
        ii = -1
        
        while t_cur <= t_fin:
  
            ii += 1
            tic = time.time()
            filename = (prefix+"%.4d")%(ii)

            if ii == 0:
                paths = path_planner.make_paths(self.X_pose)
                popt = paths[np.random.randint(len(paths))]
            else:
                if ii == 1:
                    acq = check_acquisition(acquisition, self.model, 
                                            self.inputs)
                acq.model = self.model.copy()
                acq.update_parameters()
                reward = []
                paths = path_planner.make_paths(self.X_pose)
                for p in paths:
                    qs, ts = path_planner.make_itinerary(p,200)
                    qs, ts = np.array(qs), np.array(ts)
                    ac = acq.evaluate(qs[:,0:2])
                    reward.append(np.trapz(ac.squeeze(),ts))
                popt = paths[np.argmin(reward)]

            t_int = t_cur + popt.path_length()
            time_stamps = record_time[ (record_time >  t_cur) & \
                                       (record_time <= t_int) ]

            samples = []
            for tt in time_stamps:
                samples.append(popt.sample(tt-t_cur))
            xopt = np.atleast_2d(samples)[:,0:2] 
            yopt = self.my_map.evaluate(xopt)
            t_cur = t_int

            self.X_pose = popt.path_endpoint()
            self.X = np.vstack((self.X, xopt))
            self.Y = np.vstack((self.Y, yopt))
            self.model.set_XY(self.X, self.Y)
            self.model.optimize_restarts(**kwargs_GPy)

            m_list.append(self.model.copy())
            p_list.append(popt)

            if callback:
                self._callback(ii, time.time()-tic)

            if save_iter:
                self.model.save_model(filename)

        return m_list, p_list

    @staticmethod
    def _callback(ii, time):
         m, s = divmod(time, 60)
         print("Iteration {:3d} \t Optimization completed in {:02d}:{:02d}"
               .format(ii, int(m), int(s)))
    
   
