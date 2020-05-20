import numpy as np
from ..likelihood import Likelihood


class Acquisition(object):
    """A class for unweighted acquisition functions.

    Parameters
    ----------
    model : instance of GPRegression
        A GPy model.
    inputs : instance of Inputs
        The input space.

    Attributes
    ----------
    model, inputs : see Parameters
    iter_counter : int
        Iteration counter. Gets incremented every time the method
        update_parameters() is called.

    Notes
    -----
    We follow the convention that the acquisition function is 
    to be **minimized**.

    """

    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs
        self.iter_counter = 0

    def evaluate(self, x):
        """Evaluates acquisition function at x."""
        raise NotImplementedError

    def jacobian(self, x):
        """Evaluates gradients of acquisition function at x."""
        raise NotImplementedError
   
    def update_parameters(self):
        """Update any parameter once per outer loop iteration."""
        self.iter_counter += 1

    def get_weights(self, x):
        """Likelihood ratio for unweighted acquisition functions."""
        return 1.0

    def get_weights_jac(self, x):
        """Gradients of likelihood ratio for unweighted functions."""
        return 0.0


class AcquisitionWeighted(Acquisition):
    """A class for likelihood-weighted acquisition functions.

    Parameters
    ----------
    model, inputs : see parent class (Acquisition)
    n_core : int
        Number of core iterations. If None, not used.
    likelihood : Likelihood
        A Likelihood object representing the likelihood ratio.

    Attributes
    ----------
    model, inputs, n_core : see Parameters

    """

    def __init__(self, model, inputs, likelihood=None, n_core=None):
        super(AcquisitionWeighted, self).__init__(model, inputs)
        if likelihood is None:
            likelihood = Likelihood(model, inputs)
        self.likelihood = likelihood
        self.n_core = n_core
 
    def update_parameters(self):
        self.iter_counter += 1
        if self.n_core is not None:
            if self.iter_counter <= self.n_core:
                self.likelihood.weight_type = "nominal" 
            else:
                self.likelihood.weight_type = "importance"
        self.likelihood.update_model(self.model)

    def get_weights(self, x):
        return self.likelihood.evaluate(x)

    def get_weights_jac(self, x):
        return self.likelihood.jacobian(x)


