import numpy as np
from matplotlib import pyplot as plt
from gpsearch import (Ackley, Michalewicz, Bukin, Branin, UrsemWaves, 
                      Himmelblau, Bird, RosenbrockModified, BraninModified)


def plot_likelihood_ratio(function, filename):

    my_map, inputs = function.my_map, function.inputs

    ngrid = 200
    pts = inputs.draw_samples(n_samples=ngrid, sample_method="grd")
    ndim = pts.shape[-1]
    grd = pts.reshape( (ngrid,)*ndim + (ndim,) ).T
    X, Y = grd[0], grd[1]

    # Compute map
    yy = my_map.evaluate(pts)
    ZZ = yy.reshape( (ngrid,)*ndim ).T
        
    fig = plt.figure(figsize=(1.3,1.3), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.01)

    n_contours = 7
    if filename == "ackley": n_contours = 4

    ax = plt.subplot(1, 1, 1)
    plt.contourf(X, Y, ZZ, n_contours, cmap='terrain')
    plt.contour(X, Y, ZZ, n_contours, colors='k', linewidths=0.75, linestyles="solid")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    plt.savefig(filename + "_map.pdf")
    plt.close()


if __name__ == "__main__":

    cases = [ (Ackley(), "ackley"), 
             #(Bird(), "bird"),
             #(Branin(), "branin"),
             #(BraninModified(), "braninmod"),
             #(Bukin(), "bukin"),
             #(Himmelblau(), "himmel"),
             #(Michalewicz(), "micha"),
             #(RosenbrockModified(), "rosenmod"),
             #(UrsemWaves(), "ursem")
             ]

    for (fun, fname) in cases:
        plot_likelihood_ratio(fun, fname)

