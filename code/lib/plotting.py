import string, itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.stats.proportion

def label_axes(fig_or_axes, labels=string.ascii_uppercase,
               labelstyle=r'%s',
               xy=(-0.1, 0.95), xycoords='axes fraction', **kwargs):
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
    kwargs : to be passed to annotate
             (default: ha='left', va='top', fontweight='bold)
    """
    # re-use labels rather than stop labeling
    defkwargs = dict(fontweight='bold', ha='left', va='top')
    defkwargs.update(kwargs)
    labels = itertools.cycle(labels)
    if isinstance(fig_or_axes, plt.Figure):
        axes = fig_or_axes.axes
    elif isinstance(fig_or_axes, plt.Axes):
        axes = [fig_or_axes]
    else:
        axes = fig_or_axes
    for ax, label in zip(axes, labels):
        ax.annotate(labelstyle % label, xy=xy, xycoords=xycoords,
                    **defkwargs)

# plot CDF
def plot_rankfrequency(data, ax=None,
                       normalize_x=True, normalize_y=False,
                       log_x=True, log_y=True,
                       scalex=1.0, scaley=1.0, **kwargs):
    """
    Plot rank frequency plots. 

    data: count data to be plotted
    ax: matplotlib Axes instance
    normalize_x: if True (default) plot relative frequency, if False plot raw counts
    normalize_y: if False (default) plot rank, if True plot cumulative probability
    """
    if ax is None:
        ax = plt.gca()
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if normalize_x:
        data = data/np.sum(data)
    sorted_data = np.sort(data)  # Or data.sort(), if data can be modified
    # Cumulative counts:
    if normalize_y:
        norm = sorted_data.size
    else:
        norm = 1
    ret = ax.step(sorted_data[::-1]*scalex, scaley*np.arange(sorted_data.size)/norm, **kwargs)
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    if normalize_x:
        ax.set_xlabel('Normalized clone size')
    else:
        ax.set_xlabel('Clone size')
    if not normalize_y:
        ax.set_ylabel('Clone size rank')
    return ret

def plot_insetcolorbar(vmin, vmax, cmap, step=0.1, label=None, ax=None):
    """
    Plot an inset colorbar based on a dummy axes
    """
    if ax is None:
        ax = plt.gca()
    fig, dummyax = plt.subplots()
    # make dummy plot for colorbar
    levels = np.arange(vmin, vmax+step, step)
    CS = dummyax.contourf([[0,0],[1,0]], levels, cmap=cmap)
    plt.close(fig)
    cax = inset_axes(ax, width="30%", height="3%", loc='upper right')
    cbar = plt.colorbar(CS, orientation='horizontal', cax=cax, ticks=[vmin, vmax])
    if label:
        cbar.set_label(label)
    return cax, cbar

def plot_referencescaling(ax=None, x=[4e-5, 4e-2], factor=1.0, color='k', exponent=-1.0, label=True, **kwargs):
    """
    Plot a reference power law scaling with slope -1.

    kwargs are passed to ax.plot
    """
    if ax is None:
        ax = plt.gca()
    x = np.asarray(x)
    ax.plot(x, factor*x**exponent, color=color, **kwargs)
    if label:
        xt = scipy.stats.gmean(x)
        xt = xt*1.05
        yt = factor*xt**exponent *1.05
        ax.text(xt, yt, '%g'%exponent, va='bottom', ha='left', color=color)

def statsmodels_regression(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    results = model.fit()
    return model, results

def plot_regression(x, y, ax=None,
                    logy=False, p_cutoff=0.05, fit_slope=True,
                    extend=0.0, ci=95, plot_ci=True,
                    fittype='bootstrap',
                    fittransform=None,
                    data_label='',
                    label=None,
                    **kwargs):
    """Plot a linear regression analysis.
   
    logy: log-transform data before fitting
    p_cutoff: significance cutoff for showing the fitted slope
    fit_slope: fit slope if True else rate
    fittype : in bootstrap, scipy, statsmodels
    extend: by how much to extend the fitting function beyond the fitted values
    """
    if fittype not in ['bootstrap', 'scipy', 'statsmodels']:
        raise Exception('Invalid argument')
    if label is None:
        if fittype == 'bootstrap':
            label = '{0:.0f} [{1:.0f}, {2:.0f}]'
        elif fittype == 'scipy':
            label = '${0:.0f} \pm {1:.0f}$'
        elif fittype == 'statsmodels':
            label = '${0:.0f} \pm {2:.0f} x + {1:.0f} \pm {3:.0f}$'
    if ax is None:
        ax = plt.gca()
    l, = ax.plot(x, y, 'o', label=data_label, **kwargs)
    if logy:
        y = np.log(y)

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    if fittype == 'bootstrap':
        if fit_slope:
            def robust_linregress(x, y):
                try:
                    res = scipy.stats.linregress(x, y)
                    return res
                except FloatingPointError:
                    return [np.nan]*5
            fit_parameter = 1/slope
            bootstrap = sns.algorithms.bootstrap(x, y, func=lambda x, y: 1/robust_linregress(x, y)[0], n_boot=10000)
        else:
            fit_parameter = slope
            bootstrap = sns.algorithms.bootstrap(x, y, func=lambda x, y: scipy.stats.linregress(x, y)[0], n_boot=10000)
        low, high = sns.utils.ci(bootstrap)
        print('fit', fit_parameter, 'std', np.std(bootstrap),
              'p', p_value, 'low', low, 'high', high)
        label = label.format(fit_parameter, low, high)
    elif fittype == 'scipy':
        if fit_slope:
            label = label.format(1/slope, std_err/slope**2, r_value**2)
        else:
            label = label.format(slope, std_err, r_value**2)
    elif fittype == 'statsmodels':
        x_fit = x.copy()
        if not fittransform is None:
            x_fit = fittransform(x_fit)
        model, results = statsmodels_regression(x_fit, y)
        label = label.format(results.params[1], results.params[0], results.bse[1], results.bse[0])

    print(label)

    x_fit = np.linspace(min(x)-extend, max(x)+extend, 400)
    y_fit = intercept+slope*x_fit
    if logy:
        y_fit = np.exp(y_fit)
        ax.set_yscale('log')
    ax.plot(x_fit, y_fit, c=l.get_color(),
            label=label if p_value<p_cutoff else 'NS', **kwargs)

    # error band for plot
    if plot_ci:
        def reg_func(_x, _y):
            return np.linalg.pinv(_x).dot(_y)
        X = np.c_[np.ones(len(x)), x]
        grid = np.c_[np.ones(len(x_fit)), x_fit]
        yhat = grid.dot(reg_func(X, y))
        beta_boots = sns.algorithms.bootstrap(X, y, func=reg_func,
                                    n_boot=10000).T
        yhat_boots = grid.dot(beta_boots).T
        err_bands = sns.utils.ci(yhat_boots, ci, axis=0)
        if logy:
            err_bands = np.exp(err_bands)
        ax.fill_between(x_fit, *err_bands, facecolor=l.get_color(), alpha=.3)

    return slope, intercept, r_value**2

def _split(number):
    """ Split a number in python scientific notation in its parts.
        
        @return value and exponent of number

    """
    return re.search(r'(-?[0-9].[0-9]*)(?:e\+?)(-?[0-9]*)', number).groups()

def str_quant(u, uerr, scientific=False):
    """ Make string representation in nice readable format
    
        >>> str_quant(0.0235, 0.0042, scientific = True)
        '2.4(5) \\\cdot 10^{-2}'
        >>> str_quant(1.3, 0.4)
        '1.3(4)'
        >>> str_quant(8.4, 2.3)
        '8(3)'
        >>> str_quant(-2, 0.03)
        '-2.00(3)'
	>>> str_quant(1432, 95, scientific = True)
	'1.43(10) \\\cdot 10^{3}'
	>>> str_quant(1402, 95, scientific = True)
	'1.40(10) \\\cdot 10^{3}'
        >>> str_quant(6.54, 0.14)
        '6.54(14)'
        >>> str_quant(0.8, 0.2, scientific=False)
        '0.8(2)'
        >>> str_quant(45.00, 0.05, scientific=False)
        '45.00(5)'

    """
    # preformatting
    number = format(float(u), "e")
    error = format(float(uerr), "e")
    numberValue, numberExponent = _split(number) 
    errorValue, errorExponent = _split(error)
    numberExponent, errorExponent = int(numberExponent), int(errorExponent)    

    # Precision = number of significant digits
    precision = numberExponent - errorExponent
    # make error
    if errorValue.startswith("1"):
        precision += 1
        errorValue = float(errorValue) * 10  # roundup second digit
    error = int(math.ceil(float(errorValue))) # roundup first digit

    # number digits after point (if not scientific)
    nDigitsAfterPoint = precision - numberExponent
    # make number string
    if scientific:
        number = round(float(numberValue), precision)
        if precision == 0:
            number = int(number)
    else:
        number = round(float(numberValue) * 10**numberExponent, nDigitsAfterPoint)
        if nDigitsAfterPoint == 0:
            number = int(number)
    numberString = str(number)

    # pad with 0s on right if not long enough
    if "." in numberString:
        if scientific:
            length = numberString.index(".") + precision + 1
            numberString = numberString.ljust(length, "0")
        else:
            length = numberString.index(".") + nDigitsAfterPoint + 1
            numberString = numberString.ljust(length, "0")
    
    if scientific and numberExponent != 0:
        outputString = "%s(%d) \cdot 10^{%d}" % (numberString, error, numberExponent)
    else:
        outputString = "%s(%d)" % (numberString, error)

    return outputString

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
                    xmin=None, xmax=None, step=True, scaley=1.0, **kwargs):
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
            ax.step(bins[:-1], counts*scaley, label=label, where='mid', **kwargs)
        else:
            ax.plot(0.5*(bins[:-1]+bins[1:]), counts*scaley, label=label, **kwargs)
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

