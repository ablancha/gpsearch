import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import scipy


matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 9


def latexify(ticklabels):
    """Manually set LaTeX format for tick labels."""
    return [r"$" + str(label) + "$" for label in ticklabels]


def plot_error(err_list, filename=None, logscale=True, accumulate=True,
               higher_is_better=False, tendency="median", dispersion="mad", 
               dispersion_scale=0.2, labels=None, cmap=None, xticks=None, 
               yticks=None):

    fig = plt.figure(figsize=(3.4,2.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0.02)
    ax = plt.axes()
    label = None

    for ii, err in enumerate(err_list):
        if len(err_list) == 1:
            color = 'k'
        else: 
            color = get_color(ii, cmap)
        if labels is not None:
            label = labels[ii]

        if higher_is_better:
            err = -err
        if accumulate:
            err = np.minimum.accumulate(err, axis=1)

        if err.ndim > 1:
            n_vec = np.arange(err.shape[1])
            
            if tendency == "mean":
                tend_fun = np.mean
            elif tendency == "median":
                tend_fun = np.median
            elif tendency == "gmean":
                tend_fun = scipy.stats.mstats.gmean

            if dispersion == "std":
                disp_fun = np.std
            elif dispersion == "mad":
                disp_fun = scipy.stats.median_absolute_deviation
            elif dispersion == "gstd":
                disp_fun = scipy.stats.gstd

            e_avg = tend_fun(err, axis=0)
            e_dis = disp_fun(err, axis=0)
            e_sup = e_avg + dispersion_scale * e_dis
            e_inf = e_avg - dispersion_scale * e_dis

            if logscale:
                e_sup = np.log10(e_sup)
                e_avg = np.log10(e_avg)
                e_inf = e_avg #2*e_avg - e_sup 
        #       e_sup = np.log10(e_sup)
        #       e_inf = np.log10(e_inf)

            plt.plot(n_vec, e_avg, color=color, lw=1, label=label)
            ax.fill_between(n_vec, e_inf, e_sup, color=color, alpha=0.2, lw=0)

        else:
            plt.plot(err, color=color, lw=1, label=label)

    if labels is not None:
       #plt.legend(frameon=False, loc=0, ncol=2)
        plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), 
                   loc=2, borderaxespad=0.)#, ncol=2)

    plt.xlabel('Iteration')
    plt.ylabel('Error')

    if xticks is not None:
        plt.xlim(xticks[0], xticks[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(latexify(xticks))

    if yticks is not None:
        if logscale:
            plt.ylim(10**yticks[0], 10**(yticks[-1]))
            ax.set_yticks([10**yy for yy in yticks])
            ax.set_yticklabels(latexify(["10^{"+str(yy)+"}" 
                                         for yy in yticks]))
        else:
            plt.ylim(yticks[0], yticks[-1])
            ax.set_yticks(yticks)
            ax.set_yticklabels(latexify(yticks))

    ax.tick_params(direction='in', length=2)

    if filename is None:
        filename = "err.pdf"
    plt.savefig(filename)
    plt.close()


def get_cases(cases):
    fnames, labels = map(list, zip(*cases))
    err_list = [np.load(fname + ".npy") for fname in fnames]
    return err_list, labels


def get_color(ii, cmap):
    if cmap == "cbrewer1":
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                  "#ff7f00", "#a65628", "#f781bf"]
    elif cmap == "cbrewer2":
        colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a",
                  "#e6ab02", "#a6761d", "#666666"]
    else:
        colors = ["C{}".format(i) for i in range(10)]
    return colors[ii]


