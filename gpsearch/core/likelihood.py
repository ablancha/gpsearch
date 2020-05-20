import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.mixture import GaussianMixture as GMM
from .utils import fix_dim_gmm, custom_KDE


class Likelihood(object):
    """A class for computation of the likelihood ratio.

    Parameters
    ----------
    model : instance of GPRegression
        A GPy model 
    inputs : instance of Inputs
        The input space.
    weight_type : str, optional
        Type of likelihood weight. Must be one of
            - "nominal" : uses w(x) = p(x)
            - "importance" : uses w(x) = p(x)/p_y(mu(x))
    fit_gmm : boolean, optional
        Whether or not to use a GMM approximation for the likelihood
        ratio.  
    kwargs_gmm : dict, optional
        A dictionary of keyword arguments for scikit's GMM routine.
        Use this to specify the number of Gaussian mixtures and the
        type of covariance matrix.

    Attributes
    ----------
    model, inputs, weight_type, fit_gmm, kwargs_gmm : see Parameters
    fy_interp : scipy 1-D interpolant
        An interpolant for the output pdf p_y(mu)
    gmm : scikit Gaussian Mixture Model
        A GMM object approximating the likelihood ratio.

    """

    def __init__(self, model, inputs, weight_type="importance", 
                 fit_gmm=True, kwargs_gmm=None):

        self.model = model
        self.inputs = inputs
        self.weight_type = self.check_weight_type(weight_type)
        self.fit_gmm = fit_gmm

        if kwargs_gmm is None:
            kwargs_gmm = dict(n_components=2, covariance_type="full")
        self.kwargs_gmm = kwargs_gmm

        self._prepare_likelihood()

    def update_model(self, model):
        self.model = model
        self._prepare_likelihood()
        return self

    def evaluate(self, x):
        """Evaluates the likelihood ratio at x.

        Parameters
        ----------
        x : array
            Query points. Should be of size (n_pts, n_dim)

        Returns
        -------
        w : array
            The likelihood ratio at x.

        """
        if self.fit_gmm:
            w = self._evaluate_gmm(x)
        else:
            w = self._evaluate_raw(x)
        return w

    def jacobian(self, x):
        """Evaluates the gradients of the likelihood ratio at x.

        Parameters
        ----------
        x : array
            Query points. Should be of size (n_pts, n_dim)

        Returns
        -------
        w_jac : array
            Gradients of the likelihood ratio at x.

        """
        if self.fit_gmm:
            w_jac = self._jacobian_gmm(x)
        else:
            w_jac = self._jacobian_raw(x)
        return w_jac

    def _evaluate_gmm(self, x):
        x = np.atleast_2d(x)
        w = np.exp(self.gmm.score_samples(x))
        return w[:,None]

    def _jacobian_gmm(self, x):
        x = np.atleast_2d(x)
        w_jac = np.zeros(x.shape)
        p = np.exp(self.gmm._estimate_weighted_log_prob(x))
        precisions = fix_dim_gmm(self.gmm, matrix_type="precisions")
        for ii in range(self.gmm.n_components):
            w_jac += p[:,ii,None] * np.dot(self.gmm.means_[ii]-x, \
                                           precisions[ii])
        return w_jac

    def _evaluate_raw(self, x):
        x = np.atleast_2d(x)
        fx = self.inputs.pdf(x)
        if self.weight_type == "nominal": 
            w = fx
        elif self.weight_type == "importance":
            mu = self.model.predict(x)[0].flatten()
            if self.model.normalizer:
                mu = self.model.normalizer.normalize(mu)
            fy = self.fy_interp(mu)
            w = fx/fy
        return w[:,None]

    def _jacobian_raw(self, x):
        x = np.atleast_2d(x)
        fx_jac = self.inputs.pdf_jac(x)

        if self.weight_type == "nominal":
            w_jac = fx_jac

        elif self.weight_type == "importance":
            mu = self.model.predict(x)[0].flatten()
            if self.model.normalizer:
                mu = self.model.normalizer.normalize(mu)
            mu_jac, _ = self.model.predictive_gradients(x)
            mu_jac = mu_jac[:,:,0]
            fx = self.inputs.pdf(x)
            fy = self.fy_interp(mu)
            fy_jac = self.fy_interp.derivative()(mu)
            tmp = fx * fy_jac / fy**2
            w_jac = fx_jac / fy[:,None] - tmp[:,None] * mu_jac

        return w_jac

    def _prepare_likelihood(self):
        """Prepare likelihood ratio for evaluation."""

        if self.inputs.input_dim <= 2:
            n_samples = int(1e5)
        else: 
            n_samples = int(1e6)

        pts = self.inputs.draw_samples(n_samples=n_samples, 
                                       sample_method="uni")
        fx = self.inputs.pdf(pts)

        if self.weight_type == "importance":
            mu = self.model.predict(pts)[0].flatten()
            if self.model.normalizer:
                mu = self.model.normalizer.normalize(mu)
            x, y = custom_KDE(mu, weights=fx).evaluate()
            self.fy_interp = InterpolatedUnivariateSpline(x, y, k=1)

        if self.fit_gmm:
            if self.weight_type == "nominal":
                w_raw = fx
            elif self.weight_type == "importance":
                w_raw = fx/self.fy_interp(mu)
            self.gmm = self._fit_gmm(pts, w_raw, self.kwargs_gmm)

        return self

    @staticmethod
    def _fit_gmm(pts, w_raw, kwargs_gmm):
        """Fit Gaussian Mixture Model using scikit's GMM framework.

        Parameters
        ----------
        pts : array
            Sample points. 
        w_raw : array
            Raw likelihood ratio at sample points.
        kwargs_gmm : dict
            A dictionary of keyword arguments for scikit's GMM routine.

        Returns
        -------
        gmm : scikit Gaussian Mixture Model
            A GMM object approximating the likelihood ratio.

        """
        # Sample and fit
        sca = np.sum(w_raw)
        rng = np.random.default_rng()
        aa = rng.choice(pts, size=20000, p=w_raw/sca)
        gmm = GMM(**kwargs_gmm)
        gmm = gmm.fit(X=aa)
        # Rescale
        gmm_y = np.exp(gmm.score_samples(pts))
        scgmm = np.sum(gmm_y)
        gmm.weights_ *= (sca/w_raw.shape[0] * gmm_y.shape[0]/scgmm)
        return gmm

    @staticmethod
    def check_weight_type(weight_type):
        assert(weight_type.lower() in ["nominal", "importance"])
        return weight_type.lower()


