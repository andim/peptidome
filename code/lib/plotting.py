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

