import matplotlib
from matplotlib import pyplot as plt
import numpy as np


matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 9

def latexify(ticklabels):
    """Manually set LaTeX format for tick labels."""
    return [r"$" + str(label) + "$" for label in ticklabels]


def plot_pdf(pt, pb=None, ppm=None, filename=None, 
             xticks=None, yticks=None, close=True):

    fig = plt.figure(figsize=(2.4,2.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0.02)
    ax = plt.axes()

    xt, yt = pt.evaluate()
    plt.semilogy(xt, yt, color='k', lw=1)
 
    if pb is not None:
        xb, yb = pb.evaluate()
        plt.semilogy(xb, yb, color='C0', lw=1)

    if ppm is not None:
        pm, pp = ppm
        x_min = min( pm.data.min(), pp.data.min() )
        x_max = max( pm.data.max(), pp.data.max() )
        x_eva = np.linspace(x_min - 0.1*np.abs(x_min), 
                            x_max + 0.1*np.abs(x_max), 1024)
        ym, yp = pm.evaluate(x_eva), pp.evaluate(x_eva)
        ax.fill_between(x_eva, ym, yp, \
                        color='C0', alpha=0.2, lw=0)

    plt.xlabel('$y$')
    plt.ylabel('$\mathrm{PDF}(y)$')

    if xticks is not None:
        plt.xlim(xticks[0], xticks[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(latexify(xticks))

    if yticks is not None:
        plt.ylim(10**yticks[0], 10**(yticks[-1]))
        ax.set_yticks([10**yy for yy in yticks])
        ax.set_yticklabels(latexify(["10^{"+str(yy)+"}" for yy in yticks]))

    ax.tick_params(direction='in', length=2)

    if filename is None:
        filename = "pdfs.pdf"
    plt.savefig(filename)

    if close:
        plt.close()
        

