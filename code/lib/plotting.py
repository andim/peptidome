import string, itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def label_axes(fig_or_axes, labels=string.ascii_uppercase,
               labelstyle=r'{\sf \textbf{%s}}',
               xy=(0.0, 1.0), **kwargs):
    """
    Walks through axes and labels each.
    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure or Axes to work on
    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.
    labelstyle : format string
    kwargs : to be passed to annotate (default: ha='left', va='top')
    """
    # re-use labels rather than stop labeling
    labels = itertools.cycle(labels)
    axes = fig_or_axes.axes if isinstance(fig_or_axes, plt.Figure) else fig_or_axes
    defkwargs = dict(ha='left', va='top') 
    defkwargs.update(kwargs)
    for ax, label in zip(axes, labels):
        xycoords = (ax.yaxis.label, 'axes fraction')
        ax.annotate(labelstyle % label, xy=xy, xycoords=xycoords, **defkwargs)

from scipy.interpolate import interpn

def density_scatter(x, y, ax=None, sort=True, bins=20, trans=None, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        ax = plt.gca()
    if trans is None:
        trans = lambda x: x
    data , x_e, y_e = np.histogram2d(trans(x), trans(y), bins = bins)
    z = interpn(( 0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1]) ),
                data, np.vstack([trans(x),trans(y)]).T,
                method="splinef2d", bounds_error=False )

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)
    return ax
