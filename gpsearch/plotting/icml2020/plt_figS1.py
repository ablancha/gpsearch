import scipy
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from test_funcs import (AckleyFunction, MichalewiczFunction, BukinFunction, BraninFunction, Hartmann6Function)
from gpsearch import custom_KDE, Likelihood
import GPy

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 9


def latexify(ticklabels):
    """Manually set LaTeX format for tick labels."""
    return [r"$" + str(label) + "$" for label in ticklabels]


def plot_pdf(function, n_samples, xticks, yticks, filename):

    my_map, inputs = function.my_map, function.inputs

    pts = inputs.draw_samples(n_samples=n_samples, sample_method="uni")
    yy = my_map.evaluate_parallel(pts, n_jobs=30)

    x, y = custom_KDE(yy, weights=inputs.pdf(pts)).evaluate()

    fig = plt.figure(figsize=(2.0,1.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0.02)

    # PDF plot
    ax = plt.subplot(111)
    plt.plot(x, np.log10(y), "k", lw=0.75)
    plt.xlabel(r"$y$")
    plt.ylabel(r"$\log_{10}~p_y(y)$")

    plt.xlim(xticks[0], xticks[-1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(latexify(xticks), fontsize=7)
    ax.xaxis.set_label_coords(0.5, -0.11)

    yticks = [yticks[0], 0.5*(yticks[0]+yticks[-1]), yticks[-1]]
    yticks = [ int(int(x*10.0)/10.0) for x in yticks]
    plt.ylim(yticks[0], yticks[-1])
    ax.set_yticks(yticks)
    ax.set_yticklabels(latexify(yticks), fontsize=7)

    ax.tick_params(direction='in', length=2)

    plt.savefig(filename + "_pdf.pdf")
    plt.close()


if __name__ == "__main__":

   #plot_pdf(AckleyFunction(ndim=2, noise_var=0), int(1e5), [-1,12,25], [-6,0], "ackley")
   #plot_pdf(BraninFunction(noise_var=0),         int(1e5), [-50,150,350], [-7,-1], "branin")
   #plot_pdf(BukinFunction(noise_var=0),          int(1e5), [-30,110,250], [-7,-1], "bukin")
   #plot_pdf(MichalewiczFunction(ndim=2, noise_var=0), int(2e5), [-2,-0.9,0.2], [-5,1], "micha")
   #plot_pdf(MichalewiczFunction(ndim=10, noise_var=0), int(1e6), [-7,-3,1], [-8,0], "micha10")
    plot_pdf(Hartmann6Function(noise_var=0), int(1e6), [-3.8,-1.7,0.4], [-7,1], "hart")


