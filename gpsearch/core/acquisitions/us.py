import numpy as np
from .base import Acquisition, AcquisitionWeighted


class US(Acquisition):
    """A class for Uncertainty Sampling.

    Parameters
    ----------
    model, inputs : see parent class (Acquisition)

    Attributes
    ----------
    model, inputs : see Parameters

    """

    def evaluate(self, x):
        x = np.atleast_2d(x)
        _, var = self.model.predict_noiseless(x)
        if self.model.normalizer:
            var /= self.model.normalizer.std**2
        w = self.get_weights(x)
        return - var * w

    def jacobian(self, x):
        x = np.atleast_2d(x)
        _, var = self.model.predict_noiseless(x)
        if self.model.normalizer:
            var /= self.model.normalizer.std**2
        _, var_jac = self.model.predictive_gradients(x)
        w, w_jac = self.get_weights(x), self.get_weights_jac(x)
        return - (var_jac * w + var * w_jac)


class US_LW(AcquisitionWeighted, US):
    """A class for Likelihood-Weighted Uncertainty Sampling.

    Parameters
    ----------
    model, inputs, likelihood : see parent class (AcquisitionWeighted)

    Attributes
    ----------
    model, inputs, likelihood : see Parameters

    """

    def __init__(self, model, inputs, likelihood=None):
        super().__init__(model, inputs, likelihood=likelihood)


