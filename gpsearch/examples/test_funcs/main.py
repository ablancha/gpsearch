import numpy as np
from gpsearch import OptimalDesign, regret_model, regret_obs, plot_smp
from test_funcs import Ackley, Branin, Bukin, Michalewicz


def main():

    n_init = 2
    n_iter = 50

    noise_var = 1e-3 
    b = Michalewicz(noise_var=noise_var, rescale_X=True, ndim=2)

    my_map, inputs, true_ymin, true_xmin = b.my_map, b.inputs, b.ymin, b.xmin
    X = inputs.draw_samples(n_init, "lhs")
    Y = my_map.evaluate(X)

    o = OptimalDesign(X, Y, my_map, inputs, 
                      fix_noise=False, 
                      normalize_Y=True)

    m_list = o.optimize(n_iter,
                        acquisition="PI",
                        num_restarts=20,
                        parallel_restarts=True, postpro=True)
    
    reg = regret_model(m_list, inputs, true_ymin=true_ymin, accumulate=True)

    for ii in np.arange(0, n_iter+1, 10):
        plot_smp(m_list[ii], inputs, n_init,
                 filename="smps%.4d.pdf"%(ii))

    from matplotlib import pyplot as plt
    plt.semilogy(reg)
    plt.show()

if __name__ == "__main__":
    main()


