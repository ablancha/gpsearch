import sys
sys.path.append('../../../')
from gpsearch import get_cases, get_color
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import scipy


matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.size"] = 8

def latexify(ticklabels):
    """Manually set LaTeX format for tick labels."""
    return [r"$" + str(label) + "$" for label in ticklabels]

def main():

    prefix_list = [
                   "ackley2d_noisevar1e-3_", 
                   "branin_noisevar1e-3_",
                   "bukin_noisevar1e-3_",
                   "michalewicz_noisevar1e-3_",
                  #"hartman_noisevar1e-3_"
                  ]

    metric_list = ["distmin_model", 
                   "regret_tmap"]
    metric_label = [r"$\log_{10}~d_\mathit{model}$",
                    r"$\log_{10}~r_\mathit{model}$"]

    for prefix in prefix_list:

        xticks=[0,35,70]

        if prefix == "ackley2d_noisevar1e-3_":
            filename = "fig2a.pdf"
            metric_yticks = [ [-3, -0.4], [-0.5, 1.5] ] 

        if prefix == "branin_noisevar1e-3_":
            filename = "fig2b.pdf"
            metric_yticks = [ [-2.4, -0.6], [-1.6, 1.4] ]

        if prefix == "bukin_noisevar1e-3_":
            filename = "fig2c.pdf"
            metric_yticks = [ [-1.4, -0.4], [0.0, 2.0] ]

        if prefix == "michalewicz_noisevar1e-3_":
            filename = "fig2d.pdf"
            metric_yticks = [ [-3.4, -0.4], [-3.2, 0.2] ]

        if prefix == "hartman_noisevar1e-3_":
            filename = "fig2e.pdf"
            xticks=[0,75,150]
            metric_yticks = [ [-2.2, 0.0], [-3.4, 0.4] ]

        fig = plt.figure(figsize=(3.2,2.2))#, constrained_layout=True)

        label = None

        for ii, (metric, y_label, yticks) in enumerate(zip(metric_list, metric_label, metric_yticks)):

            ax = plt.subplot(1, 2, ii+1)

            cases = [ 
                      (prefix + "EI" + "_" + metric, r"$\mathrm{EI}$"),
                      (prefix + "PI" + "_" + metric, r"$\mathrm{PI}$"),
                      (prefix + "IVR" + "_" + metric, r"$\mathrm{IVR}$"),
                      (prefix + "IVR_LW" + "_" + metric, r"$\mathrm{IVR}$-$\mathrm{LW}$"),
                      (prefix + "IVR_BO" + "_" + metric, r"$\mathrm{IVR}$-$\mathrm{BO}$"),
                      (prefix + "IVR_LWBO" + "_" + metric, r"$\mathrm{IVR}$-$\mathrm{LWBO}$"),
                      (prefix + "LCB" + "_" + metric, r"$\mathrm{LCB}$"),
                      (prefix + "LCB_LW" + "_" + metric, r"$\mathrm{LCB}$-$\mathrm{LW}$")]

            err_list, labels = get_cases(cases)


            ls = [ "solid", "solid", 
                   (0, (5, 1)), "solid", 
                   (0, (5, 1)), "solid", 
                   (0, (5, 1)), "solid"]

            colors = ["#a65628", "#999999",
                      "#4daf4a", "#4daf4a",
                      "#377eb8", "#377eb8",
                      "#e41a1c", "#e41a1c"] 

            plot_error(err_list, ax, colors=colors, ls=ls, y_label=y_label, labels=labels, 
                       dispersion_scale=0.25, xticks=xticks, yticks=yticks)


            if ii == 0 :
                ax.legend(frameon=False, bbox_to_anchor=(-0.32, 1.05, 2.8, 1.2), 
                          loc='lower left',
                          ncol=4, mode="expand", borderaxespad=0)

        plt.subplots_adjust(wspace=0.4, hspace=0.0, 
                            left=0.12, right=0.98, 
                            top=0.8, bottom=0.12)

        plt.savefig(filename)


def plot_error(err_list, ax, colors=None, ls=None, y_label=None, labels=None, 
               dispersion_scale=0.2, xticks=None, yticks=None):


    for ii, err in enumerate(err_list):
        
        label = labels[ii]
        lstyle = ls[ii]
        color = colors[ii]

        err = np.minimum.accumulate(err, axis=1)

        n_vec = np.arange(err.shape[1])
        tend_fun = np.median
        disp_fun = scipy.stats.median_absolute_deviation

        e_avg = tend_fun(err, axis=0)
        e_dis = disp_fun(err, axis=0)
        e_sup = e_avg + dispersion_scale * e_dis

        e_sup = np.log10(e_sup)
        e_avg = np.log10(e_avg)
        e_inf = e_avg #2*e_avg - e_sup 

        ax.plot(n_vec, e_avg, color=color, linestyle=lstyle, lw=0.75, label=label)
        ax.fill_between(n_vec, e_inf, e_sup, color=color, alpha=0.1, lw=0)

    plt.xlabel(r"$\mathrm{Iteration}$")
    plt.ylabel(y_label)

    plt.xlim(xticks[0], xticks[-1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(latexify(xticks), fontsize=7)
    ax.xaxis.set_label_coords(0.5, -0.11)

    yticks = [yticks[0], 0.5*(yticks[0]+yticks[-1]), yticks[-1]]
    yticks = [ int(x*10.0)/10.0 for x in yticks]
    plt.ylim(yticks[0], yticks[-1])
    ax.set_yticks(yticks)
    ax.set_yticklabels(latexify(yticks), fontsize=7)
    ax.yaxis.set_label_coords(-0.23, 0.5)

    ax.tick_params(direction='in', length=2)




if __name__ == "__main__":
    main()


