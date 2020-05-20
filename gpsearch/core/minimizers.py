import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed


def funmin(fun, jac, inputs, opt_method="l-bfgs-b", args=(), 
           kwargs_op=None, num_restarts=None, parallel_restarts=False, 
           n_jobs=10, init_method=None):
    """Scipy-based minimizer allowing multiple parallel restarts.

    Parameters
    ----------
    fun : callable
        Objective function to be minimized.
    jac : callable
        Jacobian of the objective function.
    inputs : instance of Inputs
        To be used for domain definition.
    opt_method : str, optional 
        Type of solver. Should be one of "L-BFGS-B", "SLSQP" or "TNC".
    args : tuple, optional
        Extra arguments passed to fun and jac.
    kwargs_op : dict, optional
        A dictionary of solver options, as in scipy.optimize.minimize.
    num_restarts : int, optional
        Number of restarts for the optimizer. The number of initial
        guesses is 1+num_restarts. If None, min(100,10*d) is supplied, 
        with d the dimension of the input space.
    parallel_restarts : boolean, optional
        Whether or not to solve the optimization problems in parallel.
    n_jobs : int, optional
        Number of workers used by joblib for parallel computation.
    init_method : str, optional
        Sampling method for initial guesses. If None, the points are 
        selected by Latin-Hypercube Sampling.  If "sample_fun", we draw
        1000 uniformly sampled points and retain those with the smallest
        objective value. This approach can help avoid local minima, but 
        it is computationaly more expensive.

    Returns
    -------
    x_opt : array
        The solution array.  For multiple initial guesses, the solution
        array associated with the smallest objective value is returned.

    """
    opt_method = opt_method.lower()
    assert(opt_method in ["l-bfgs-b", "slsqp", "tnc"])

    if kwargs_op is None:
        kwargs_op = dict(options={"disp":False})

    if num_restarts is None:
        num_restarts = min(100, 10*inputs.input_dim)

    n_guess = num_restarts + 1

    if init_method is None:
        x0 = inputs.draw_samples(n_guess, "lhs")

    elif init_method == "sample_fun":
        X0 = inputs.draw_samples(1000, "uni")
        scores = fun(np.atleast_2d(X0), *args)
        sorted_idxs = np.argsort(scores)
        x0 = X0[sorted_idxs[:min(len(scores), n_guess)], :]

    if parallel_restarts:
        res = Parallel(n_jobs=n_jobs, backend="loky")(
                       delayed(minimize)(fun, 
                                         np.atleast_2d(x0[i]), 
                                         args=args,
                                         method=opt_method, 
                                         jac=jac,
                                         bounds=inputs.domain,
                                         **kwargs_op)
                       for i in range(x0.shape[0]) )

    else:
        res = [ minimize(fun, 
                         np.atleast_2d(x0[i]), 
                         args=args, 
                         method=opt_method,
                         jac=jac, 
                         bounds=inputs.domain,
                         **kwargs_op)
                for i in range(x0.shape[0]) ]

    idx = np.argmin([r.fun for r in res])
    xopt = res[idx].x

    return xopt


