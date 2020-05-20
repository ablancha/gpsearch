import unittest
import numpy as np
import GPy
from GPy.models import GradientChecker
from gpsearch import (RBF, UniformInputs, Likelihood, EI, PI,\
                       LCB, LCB-LW, US, US_LW, IVR, IVR-LW)


class MiscTests(unittest.TestCase):
    def setUp(self):
        self.N = 20
        self.N_new = 50
        self.D = 1
        self.X = np.random.uniform(-3., 3., (self.N, 1))
        self.Y = np.sin(self.X) + np.random.randn(self.N, self.D) * 0.05
        self.X_new = np.random.uniform(-3., 3., (self.N_new, 1))

    def test_setXY(self):
        m = GPy.models.GPRegression(self.X, self.Y)
        m.set_XY(np.vstack([self.X, np.random.rand(1,self.X.shape[1])]), np.vstack([self.Y, np.random.rand(1,self.Y.shape[1])]))
        m._trigger_params_changed()
        self.assertTrue(m.checkgrad())
        m.predict(m.X)


if __name__ == "__main__":
    print("Running unit tests, please be (very) patient...")
    unittest.main()


