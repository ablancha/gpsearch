import numpy as np
from scipy.stats import norm, lognorm, uniform
from gpsearch import (BlackBox, OptimalDesign, Inputs,
                      Benchmarker, BenchmarkerLHS, custom_KDE)

def restore_X(xsc):
    domain = [ [0.05,0.15], [100,50000], [63070,115600], [990,1110],
               [63.1,116], [700,820], [1120,1680], [9855,12045] ]
    xsc_shape = xsc.shape
    xsc = np.atleast_2d(xsc)
    x = np.zeros(xsc.shape)
    for i in range(x.shape[1]):
        bd = domain[i]
        x[:,i] = xsc[:,i]*(bd[1]-bd[0]) + bd[0]
    return x.reshape(xsc_shape)


def map_def(x):
    x = restore_X(x)
    num = 2*np.pi*x[2]*(x[3]-x[5])
    c = np.log(x[1]/x[0])
    den = c * ( 1 + 2*x[6]*x[2]/(c*x[0]**2*x[7]) + x[2]/x[4] )
    return num / den


class CustomInputs(Inputs):

    def pdf(self, x):
        x = np.atleast_2d(restore_X(x))
        pdf = norm.pdf(x[:,0], loc=0.1, scale=0.0161812) \
            * lognorm.pdf(x[:,1], s=1.0056, scale=np.exp(7.71))

        lb, ub = map(list, zip(*self.domain))
        lb, ub = restore_X(np.array(lb)), restore_X(np.array(ub))

        for ii in range(2, self.input_dim):
            bd = self.domain[ii]
            pdf *= uniform.pdf(x[:,ii], lb[ii], ub[ii]-lb[ii])
        return pdf / np.prod(ub-lb)


def main():

    ndim = 8 
    noise_var = 1e-2 
    my_map = BlackBox(map_def, noise_var=noise_var)
    inputs = CustomInputs([ [0,1] ] * ndim)
    
    np.random.seed(3)

    filename = "map_samples{:d}D.txt".format(ndim)
    try:
        smpl = np.genfromtxt(filename)
        pts = smpl[:,0:-1]
        yy = smpl[:,-1]
    except:
        pts = inputs.draw_samples(n_samples=int(1e6), sample_method="uni")
        yy = my_map.evaluate(pts, parallel=True, include_noise=False)
        np.savetxt(filename, np.column_stack((pts,yy)))
    weights = inputs.pdf(pts)
    pt = custom_KDE(yy, weights=weights)

    prefix = "gmm2_noise2"

    n_init = ndim + 1
    n_iter = 160
    n_trials = 100
    n_jobs = 15

    metric = [ ("log_pdf", dict(pt=pt, pts=pts)) ]
    acq_list = ["US", "US_LW", 
                "IVR", "IVR_IW", "IVR_LW"]

    for acq in acq_list:
        print("Benchmarking " + acq)
        b = Benchmarker(my_map, acq, n_init, n_iter, inputs, metric)
        result = b.run_benchmark(n_trials, n_jobs, filename=prefix+acq)

    b = BenchmarkerLHS(my_map, n_init, n_iter, inputs, metric)
    result = b.run_benchmark(n_trials, n_jobs, filename=prefix+"LHS")


if __name__ == "__main__":
    main()


