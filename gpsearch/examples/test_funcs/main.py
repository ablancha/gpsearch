import numpy as np
from gpsearch import (OptimalDesign, plot_smp, 
                      regret_model, regret_obs, 
                      distmin_model, distmin_obs)
from gpsearch.core.acquisitions import *
from test_funcs import (AckleyFunction, BraninFunction, 
                        BukinFunction, Hartmann6Function,
                        MichalewiczFunction)


def main():

    n_init = 5
    n_iter = 50

    noise_var = 1e-2 * 0
    b = MichalewiczFunction(noise_var=noise_var, rescale_X=True, ndim=2)

    my_map, inputs, true_ymin, true_xmin = b.my_map, b.inputs, b.ymin, b.xmin
    X = inputs.draw_samples(n_init, "lhs")
    Y = my_map.evaluate(X)

    o = OptimalDesign(X, Y, my_map, inputs, 
                      fix_noise=False, 
                      normalize_Y=True)

    acquisition = IVR_LWBO(o.model, o.inputs)
    m_list = o.optimize(n_iter,
                        acquisition=acquisition,
                        num_restarts=20,
                        parallel_restarts=True)
    exit()
    
    reg = regret_model(m_list, inputs, true_ymin=true_ymin)
    reo = regret_obs(m_list, inputs, true_ymin=true_ymin)
    dis = distmin_model(m_list, inputs, true_xmin=true_xmin)
    dio = distmin_obs(m_list, inputs, true_xmin=true_xmin)

    from matplotlib import pyplot as plt
    plt.semilogy(reg);
    plt.semilogy(reo, '--');
    plt.semilogy(dis); 
    plt.semilogy(dio, '--') 

    plt.show();  exit()

    for ii in np.arange(0, n_iter+1, 10):
        plot_smp(m_list[ii], inputs, n_init,
                 filename="smps%.4d.pdf"%(ii))

if __name__ == "__main__":
    main()


