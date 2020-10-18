import numpy as np
from matplotlib import pyplot as plt
from gpsearch import (BlackBox, UniformInputs, Michalewicz, OptimalPath, PathPlanner)
import dubins


def main():

    np.random.seed(2)

    noise_var = 1e-3
    b = Michalewicz(noise_var=noise_var, rescale_X=True)
    my_map, inputs, true_ymin, true_xmin = b.my_map, b.inputs, b.ymin, b.xmin

    X_pose = (0,0, np.pi/4)
    Y_pose = my_map.evaluate(X_pose[0:2])

    p = PathPlanner(inputs.domain, 
                    look_ahead=0.2, 
                    turning_radius=0.02)
    record_time = np.linspace(0,5,51)
    o = OptimalPath(X_pose, Y_pose, my_map, inputs)
    m_list, p_list = o.optimize(record_time, 
                                path_planner=p, 
                                acquisition="US")

    # Compute true map
    ngrid = 50
    pts = inputs.draw_samples(n_samples=ngrid, sample_method="grd")
    ndim = pts.shape[-1]
    grd = pts.reshape( (ngrid,)*ndim + (ndim,) ).T
    X, Y = grd[0], grd[1]
    yy = my_map.evaluate(pts, include_noise=False)
    ZZ = yy.reshape( (ngrid,)*ndim ).T 
    plt.contourf(X, Y, ZZ); 

    # Plot UAV trajectory
    r_list = [ np.array(p.make_itinerary(path,1000)[0]) for path in p_list ]
    for ii in np.arange(0, len(m_list), 5):
        model = m_list[ii]
        yy = model.predict(pts)[0]
        ZZ = yy.reshape( (ngrid,)*ndim ).T
        plt.contourf(X, Y, ZZ);
        for rr in r_list[0:ii+1]:
            plt.plot(rr[:,0], rr[:,1], 'r-')
        plt.plot(model.X[:,0], model.X[:,1], 'ro')
        plt.axis('equal')
        plt.show()


if __name__ == "__main__":
    main()



