import numpy as np
from gpsearch import BlackBox, UniformInputs

class TestFunction():
    """
    This wrapper takes another objective and scales its input domain 
    to [0,1]^d. 
    """
    def __init__(self, function, domain, ymin, xmin, args={}, kwargs={},
                 noise_var=0.0, rescale_X=False):
        self.ymin = ymin
        self.bounds = domain
        self.function = function
 
        if rescale_X:
            self.xmin = self.rescale_X(xmin)
            domain = [ [0,1] ] * len(domain) 
            self.my_map = BlackBox(self.function_sc, args=args, kwargs=kwargs,
                               noise_var=noise_var)
        else:
            self.xmin = xmin
            self.my_map = BlackBox(self.function, args=args, kwargs=kwargs,
                               noise_var=noise_var)

        self.inputs = UniformInputs(domain)

    def function_sc(self, x, *args, **kwargs):
        x = self.restore_X(x).reshape(x.shape)
        return self.function(x, *args, **kwargs)

    def rescale_X(self, x):
        x = np.atleast_2d(x)
        xsc = np.zeros(x.shape)
        for i in range(xsc.shape[1]):
            bd = self.bounds[i]
            xsc[:,i] = ( x[:,i] - bd[0] ) / ( bd[1] - bd[0] )
        return xsc

    def restore_X(self, xsc):
        xsc = np.atleast_2d(xsc)
        x = np.zeros(xsc.shape)
        for i in range(x.shape[1]):
            bd = self.bounds[i]
            x[:,i] = xsc[:,i]*(bd[1]-bd[0]) + bd[0]
        return x
        

class Ackley(TestFunction):

    def __init__(self, noise_var=0.0, rescale_X=False, ndim=2):

        function = self._ackley
        domain = [ [-32.768, 32.768] ] * ndim
        y_min = 0.0
        x_min = [ [0] * ndim ]

        if ndim == 2:
            std = 2.4
        elif ndim == 5:
            std = 0.8
        elif ndim == 10:
            std = 0.4

        noise_var = noise_var * std**2

        super().__init__(function, domain, y_min, x_min, args=(ndim,), 
                         rescale_X=rescale_X, noise_var=noise_var) 

    @staticmethod
    def _ackley(x, ndim=2):
        a = 20
        b = 0.2
        c = 2*np.pi
        return - a * np.exp( -b * np.sqrt(np.sum(x**2)/ndim) ) \
               - np.exp( np.sum(np.cos(c*x))/ndim ) \
               + a + np.exp(1.0)
        

class Bird(TestFunction):

    def __init__(self, noise_var=0.0, rescale_X=False):

        function = self._bird
        domain = [ [-2*np.pi, 2*np.pi] ] * 2
        y_min = -106.764537
        x_min = [ [4.70104, 3.15294], [-1.58214, -3.13024] ]

        std = 42.4
        noise_var = noise_var * std**2

        super().__init__(function, domain, y_min, x_min,
                         rescale_X=rescale_X, noise_var=noise_var)

    @staticmethod
    def _bird(x):
        x1, x2 = x[0], x[1]
        y = np.sin(x1) * np.exp( (1-np.cos(x2))**2 ) \
          + np.cos(x2) * np.exp( (1-np.sin(x1))**2 ) + (x1-x2)**2 
        return y

        
class Branin(TestFunction):
    
    def __init__(self, noise_var=0.0, rescale_X=False):

        function = self._branin
        domain = [ [-5,10], [0,15] ]
        y_min = 0.397887 
        x_min = [ [-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475] ]

        std = 51.2
        noise_var = noise_var * std**2

        super().__init__(function, domain, y_min, x_min, 
                         rescale_X=rescale_X, noise_var=noise_var) 

    @staticmethod
    def _branin(x):
        a = 1.0
        b = 5.1 / (4*np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8*np.pi)
        x1, x2 = x[0], x[1]
        y = a * (x2 - b*x1**2 + c*x1 -r)**2 + s * (1-t) * np.cos(x1) + s
        return y


class BraninModified(TestFunction):

    def __init__(self, noise_var=0.0, rescale_X=False):

        function = self._braninmodified
        domain = [ [-5,10], [0,15] ]
        y_min = -0.179891239069905
        x_min = [ [-3.196988423389338,12.526257883092258] ]

        std = 0.02817
        noise_var = noise_var * std**2

        super().__init__(function, domain, y_min, x_min,
                         rescale_X=rescale_X, noise_var=noise_var)

    @staticmethod
    def _braninmodified(x):
        a = 1.0
        b = 5.1 / (4*np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8*np.pi)
        x1, x2 = x[0], x[1]
        f1 = a * (x2 - b*x1**2 + c*x1 -r)**2
        f2 = s * (1-t) * np.cos(x1) * np.cos(x2)
        f3 = np.log(x1**2 + x2**2 + 1)
        y = -1/(f1+f2+f3+s)
        return y


class Bukin(TestFunction):

    def __init__(self, noise_var=0.0, rescale_X=False):

        function = self._bukin
        domain = [ [-15,-5], [-3,3] ]
        y_min = 0.0
        x_min = [ [-10,1] ]

        std = 45.3
        noise_var = noise_var * std**2

        super().__init__(function, domain, y_min, x_min,
                         rescale_X=rescale_X, noise_var=noise_var)

    @staticmethod
    def _bukin(x):
        x1, x2 = x[0], x[1]
        y = 100 * np.sqrt( np.abs(x2-0.01*x1**2) ) + 0.01 * np.abs(x1+10)
        return y


class Hartmann6(TestFunction):

    def __init__(self, noise_var=0.0, rescale_X=False):

        function = self._hartmann6
        domain = [ [0,1] ] * 6
        y_min = -3.32237
        x_min = [ [0.20169, 0.150011, 0.476874,
                   0.275332, 0.311652, 0.6573] ]

        std = 0.38
        noise_var = noise_var * std**2

        super().__init__(function, domain, y_min, x_min,
                         rescale_X=rescale_X, noise_var=noise_var)

    @staticmethod
    def _hartmann6(x):

        alpha = np.array([1.0, 1.2, 3.0, 3.2])

        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])

        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091, 381]])

        arg = np.dot(A, (x-P).T**2) 
        y = -np.dot(alpha, np.diag(np.exp(-arg)))

        return y


class Himmelblau(TestFunction):

    def __init__(self, noise_var=0.0, rescale_X=False):

        function = self._himmelblau
        domain = [ [-6, 6] ] * 2
        y_min = 0
        x_min = [ [3,2], [-2.805118,3.283186], [-3.779310,-3.283186],
                  [3.584458,-1.848126] ]

        std = 282.4
        noise_var = noise_var * std**2

        super().__init__(function, domain, y_min, x_min,
                         rescale_X=rescale_X, noise_var=noise_var)

    @staticmethod
    def _himmelblau(x):
        x1, x2 = x[0], x[1]
        y = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
        return y


class Michalewicz(TestFunction):

    def __init__(self, noise_var=0.0, rescale_X=False, ndim=2):

        function = self._michalewicz
        domain = [ [0, np.pi ] ] * ndim

        if ndim == 2:
            y_min = -1.8013
            x_min = [ [2.2, 1.57] ]
            std = 0.32
        elif ndim == 5:
            y_min = -4.6877
            x_min = [ [2.29, 1.57, 1.28, 1.92, 1.72] ]
            std = 0.51
        elif ndim == 10:
            y_min = -9.6602
            x_min = [ [2.29, 1.57, 1.28, 1.92, 1.72, 
                       1.57, 1.45, 1.75, 1.65, 1.57] ]
            std = 0.72

        noise_var = noise_var * std**2

        super().__init__(function, domain, y_min, x_min, args=(ndim,),
                         rescale_X=rescale_X, noise_var=noise_var)

    @staticmethod
    def _michalewicz(x, ndim=2):
        m = 10
        y = 0
        for i in range(ndim):
            y += np.sin(x[i]) * np.sin( (i+1) * x[i]**2 / np.pi )**(2*m)
        return -y


class OakleyOHagan(TestFunction):

    def __init__(self, noise_var=0.0, rescale_X=False):

        function = self._oakleyohagan
        domain = [ [-4,4] ]*2

        y_min = 5 - 2*np.sqrt(3) - 11*np.pi/6
        x_min = [[-7*np.pi/6, -2*np.pi/3]]
        std = 4.04738

        noise_var = noise_var * std**2

        super().__init__(function, domain, y_min, x_min,
                         rescale_X=rescale_X, noise_var=noise_var)

    @staticmethod
    def _oakleyohagan(x):
        y = 5 + x[0] + x[1] + 2*np.cos(x[0]) + 2*np.sin(x[1]) 
        return y


class RosenbrockModified(TestFunction):

    def __init__(self, noise_var=0.0, rescale_X=False):

        function = self._rosenbrockmodified
        domain = [[-1.0, 0.5], [-1.0, 1.0]]

        y_min = 34.0402431
        x_min = [[-0.90955374, -0.95057172]]
        std = 36.3989

        noise_var = noise_var * std**2

        super().__init__(function, domain, y_min, x_min,
                         rescale_X=rescale_X, noise_var=noise_var)

    @staticmethod
    def _rosenbrockmodified(x):
        y = 74 + 100. * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        y -= 400 * np.exp(-((x[0] + 1.) ** 2 + (x[1] + 1.) ** 2) / 0.1)
        return y


class UrsemWaves(TestFunction):

    def __init__(self, noise_var=0.0, rescale_X=False):

        function = self._ursemwaves
        domain = [(-0.9, 1.2), (-1.2, 1.2)]

        y_min = -8.5536
        x_min = [ [1.2,1.2] ]
        std = 2.75049

        noise_var = noise_var * std**2

        super().__init__(function, domain, y_min, x_min, 
                         rescale_X=rescale_X, noise_var=noise_var)

    @staticmethod
    def _ursemwaves(x):
        x1, x2 = x[0], x[1] 
        u = -0.9*x1**2
        v = (x2**2 - 4.5*x2**2) * x1 * x2
        w = 4.7*np.cos(3*x1 - x2**2 * (2+x1)) * np.sin(2.5*np.pi*x1)
        return u + v + w



