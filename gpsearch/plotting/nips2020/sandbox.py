import numpy as np
from matplotlib import pyplot as plt
from gpsearch import (BlackBox, PathPlanner)
from gpsearch import Branin
import dubins


def main():

    np.random.seed(2)

    noise_var = 1e-3 * 0
    b = Branin(noise_var=noise_var, rescale_X=True)
    my_map, inputs, true_ymin, true_xmin = b.my_map, b.inputs, b.ymin, b.xmin

    ngrid = 50
    pts = inputs.draw_samples(n_samples=ngrid, sample_method="grd")
    ndim = pts.shape[-1]
    grd = pts.reshape( (ngrid,)*ndim + (ndim,) ).T
    X, Y = grd[0], grd[1]

    # Compute map
    yy = my_map.evaluate(pts)
    ZZ = yy.reshape( (ngrid,)*ndim ).T 
    plt.contourf(X, Y, ZZ); 

    X = inputs.draw_samples(4, "lhs")

    qq = [ (0.0   , 0.0   , np.pi/4),
           (X[0,0], X[0,1], np.pi/2),
           (X[1,0], X[1,1], np.pi/3)
         ]

   #q0 = (X[0,0], X[0,1], np.pi/2)
   #q0 = (0.0,0.0, np.pi/4)
   #q1 = (X[1,0], X[1,1], np.pi/3)
   #q1 = (0.5,0.5, np.pi/2)

    fov = np.pi/3
    look_ahead = 0.3 
    turning_radius = 0.1 
    step_size = 0.001

    pp = PathPlanner(inputs.domain, 
                     padding=0,
                     n_frontier=400,
                     look_ahead=look_ahead, 
                     turning_radius=turning_radius, 
                     fov=fov)

    q0 = qq[0]

    idx_f = [220, 50, 250, 350]
    idx_f = [180, 250, 20]

    for (ii, idx) in enumerate(idx_f):
        frontier = pp.make_frontier(q0)
        ff = np.array(frontier)
   
        print(ff)
        q1 = ff[idx]

        path = dubins.shortest_path(q0, q1, turning_radius)
        qs, _ = path.sample_many(step_size)
        qs.append(path.path_endpoint())
        qs = np.atleast_2d(qs)
        plt.plot(qs[:,0], qs[:,1], 'r-')

        q0 = q1


   #for (q0,q1) in zip(qq,qq[1:]):

    frontier = pp.make_frontier(q1)
    ff = np.array(frontier)

#   rr = np.array(it[2])
#   plt.plot(rr[:,0], rr[:,1], 'c-')
    plt.plot(ff[:,0], ff[:,1], 'y')
    plt.axis('equal')
    plt.show() 
    exit()
    


if __name__ == "__main__":
    main()



