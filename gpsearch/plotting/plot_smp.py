import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 9

def latexify(ticklabels):
    """Manually set LaTeX format for tick labels."""
    return [r"$" + str(label) + "$" for label in ticklabels]



def plot_smp(model, inputs, n_init, *args, **kwargs):
    if model.input_dim == 2:
        plot_smp2D(model, inputs, n_init, *args, **kwargs)
    elif model.input_dim == 3:
        plot_smp3D(model, inputs, n_init, *args, **kwargs)
    else:
        raise NotImplementedError("Plotting in more than three " \
                                  + "dimensions not supported")


def plot_smp2D(model, inputs, n_init, filename=None, 
               xticks=None, yticks=None, cmapticks=None, close=True):

    fig = plt.figure(figsize=(2.4,2.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
    ax = plt.axes()

    pts = inputs.draw_samples(n_samples=100, sample_method="grd")
    mu = model.predict(pts)[0].flatten()

    if cmapticks is not None:
        sc = plt.scatter(pts[:,0], pts[:,1], c=mu,  
                         vmin=cmapticks[0], vmax=cmapticks[-1])
        cbar = plt.colorbar(sc, ticks=cmapticks)
        cbar.ax.set_yticklabels(latexify(cmapticks))
    else:
        sc = plt.scatter(pts[:,0], pts[:,1], c=mu)
        cbar = plt.colorbar(sc)
    cbar.ax.tick_params(direction='in', length=2)

    X1 = model.X[:,0]
    X2 = model.X[:,1]
    plt.plot(X1[0:n_init], X2[0:n_init], '^', markersize=4,  
             markerfacecolor="white", markeredgecolor="k", 
             markeredgewidth=0.5)
    plt.plot(X1[n_init::], X2[n_init::], 'o', markersize=4,  
             markerfacecolor="violet", markeredgecolor="k", 
             markeredgewidth=0.5)

    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')

    lb, ub = map(list, zip(*inputs.domain))
    plt.xlim(lb[0], ub[0])
    plt.ylim(lb[1], ub[1])

    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(latexify(xticks))

    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(latexify(yticks))
    ax.tick_params(direction='in', length=2)

    if filename is None:
        filename = "smps.pdf"
    plt.savefig(filename)

    if close:
        plt.close()


def plot_smp3D(model, inputs, n_init, filename, \
               xticks=None, yticks=None, zticks=None, close=True):

    fig = plt.figure(figsize=(2.4,2.2))#, constrained_layout=True)
   # fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
    ax = plt.axes(projection='3d')
    ax.grid(False)
    ax.view_init(elev=25, azim=135)

    X1 = model.X[:,0]
    X2 = model.X[:,1]
    X3 = model.X[:,2]

    ax.scatter(X1[0:n_init], X2[0:n_init], X3[0:n_init], marker='^', s=4, \
             c="white", edgecolors="k", linewidths=0.5, alpha=1)
    ax.scatter(X1[n_init::], X2[n_init::], X3[n_init::], marker='o', s=4, \
             c="violet", edgecolors="k", linewidths=0.5, alpha=1)

    ax.set_xlabel('')#r'$\theta_1$')
    ax.set_ylabel('')#r'$\theta_2$')
    ax.set_zlabel('')#r'$\theta_3$')
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_zlim(zticks[0], zticks[-1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)
    ax.set_xticklabels('')#latexify(xticks), va='center', ha='right')
    ax.set_yticklabels('')#latexify(yticks), va='center', ha='left')
    ax.set_zticklabels('')#latexify(zticks), va='center', ha='left')

    for aa in [ax.xaxis, ax.yaxis, ax.zaxis]:
        aa.pane.set_edgecolor("black")
        aa.pane.set_alpha(1)
        aa.pane.fill = False
        aa._axinfo['tick']['inward_factor'] = 0
        aa._axinfo['tick']['outward_factor'] = 0.4
    #   aa._axinfo['label']['space_factor'] = -12.8

    fig.tight_layout()
    plt.savefig(filename)
    if close:
        plt.close()



class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

