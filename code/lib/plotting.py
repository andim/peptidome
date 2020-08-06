import string, itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.stats.proportion

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
    x = np.asarray(x)
    y = np.asarray(y)
    if ax is None :
        ax = plt.gca()
    if trans is None:
        trans = lambda x: x
    data , x_e, y_e = np.histogram2d(trans(x), trans(y), bins=bins)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])),
                data, np.vstack([trans(x),trans(y)]).T,
                method="splinef2d", bounds_error=False)

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    scatter_kwargs = dict(rasterized=True)
    scatter_kwargs.update(kwargs)
    ax.scatter(x, y, c=z, **scatter_kwargs)
    return ax

def plot_histograms(valuess, labels, weights=None, nbins=40, ax=None,
                    xmin=None, xmax=None, step=True, **kwargs):
    if not ax:
        ax = plt.gca()
    if (xmin is None) or (xmax is None):
        mean = np.mean([np.mean(values) for values in valuess])
        std = np.mean([np.std(values) for values in valuess])
    if xmin is None:
        xmin = round(mean-5*std)
    if xmax is None:
        xmax  = round(mean+5*std)
    bins = np.linspace(xmin, xmax, nbins)
    for i, (values, label) in enumerate(zip(valuess, labels)):
        # filter nans
        values = np.asarray(values)
        mask = ~np.isnan(values)
        values = values[mask]
        if (not weights is None) and (not weights[i] is None):
            weight = weights[i][mask]
            counts, bins = np.histogram(values, bins=bins, weights=weight)
            counts /= np.sum(weight)
        else:
            counts, bins = np.histogram(values, bins=bins)
            counts = counts/np.sum(counts)
        if step:
            ax.step(bins[:-1], counts, label=label, where='mid', **kwargs)
        else:
            ax.plot(0.5*(bins[:-1]+bins[1:]), counts, label=label, **kwargs)
    ax.legend()
    return ax


def plot_proportion(x, counts, totalcounts, label=None, ax=None, **kwargs):
    """ Plot proportions with errorbars.

    counts : positive counts in x-bin
    totalcounts : all counts in x-bin
    """
    if ax is None:
        ax = plt.gca()
    n = totalcounts
    mask = n>0
    proportion = (counts+0.5)/(n+1)
    l, = ax.plot(x[mask], proportion[mask], label=label, **kwargs)
    lower, upper = statsmodels.stats.proportion.proportion_confint(counts, n, alpha=0.05, method='beta')
    ax.fill_between(x[mask], lower[mask], upper[mask], alpha=.5)
    return l

