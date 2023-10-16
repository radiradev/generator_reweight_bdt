import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# Define default plot styles
plot_style_0 = {
    'histtype': 'step',
    'color': 'black',
    'linewidth': 2,
    'linestyle': '--'
}
plot_style_1 = {'alpha': 0.5}

# Absolute plotting params
rc('font', family='serif')
rc('font', size=22)
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
rc('legend', fontsize=15)


def probability_plots(probas_a, probas_b, generator_a=None, generator_b=None):
    fig, axs = plt.subplots(2, 2)

    # NN Probability plot

    if generator_a is None:
        generator_a = 'Origin'
    if generator_b is None:
        generator_b = 'Target'

    counts_ref, bins_ref, _ = axs[0, 0].hist(probas_a,
                                             histtype='step',
                                             bins=100,
                                             label=generator_a,
                                             linewidth=2)
    counts, bins, _ = axs[0, 0].hist(probas_b,
                                     histtype='step',
                                     bins=100,
                                     label=generator_b,
                                     linewidth=2)
    axs[0, 0].legend()

    # Probability calibration plot
    x_values = [
        0.5 * (bins_ref[i + 1] + bins_ref[i]) for i in range(len(counts_ref))
    ]

    y_values = 1 - counts_ref / (counts_ref + counts)
    y_values_ref = np.arange(0, 1, 1 / len(counts_ref))
    axs[0, 1].plot(x_values,
                   y_values_ref,
                   '--',
                   linewidth=2,
                   color='tab:blue',
                   alpha=1,
                   label='reference')
    axs[0, 1].plot(x_values,
                   y_values,
                   '-',
                   color='tab:orange',
                   linewidth=2,
                   label='probas')
    axs[0, 1].legend()

    # Probability errors
    y_error = y_values - y_values_ref
    x_values = np.arange(len(y_values))
    axs[1, 1].plot(x_values,
                   np.zeros_like(x_values),
                   '--',
                   label='reference',
                   linewidth=2)
    axs[1, 1].plot(x_values, y_error, 'o', label='probas error')
    axs[1, 1].legend()
    plt.suptitle('Probability Output')


def kl_divergence(p, q):
    return np.sum(np.where(p != 0.0, p * np.log(p / q), 0.0))


def plot_2d_hist(x_a, y_a, x_b, y_b, weights, bins, bin_range, xlabel, ylabel, density=True, cmap='jet'):

    # orgnaize the plots using subplots
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))

    # nominal distribution
    h_a, xedges, yedges, hist_a = axs[0, 0].hist2d(x_a, y_a, bins=bins, range=bin_range, cmap=cmap, density=density)
    axs[0, 0].set_title('10a')

    # Save common vmin and vmax
    vmin_hist = hist_a.norm.vmin
    vmax_hist = hist_a.norm.vmax

    # target distribution
    h_b, xedges_b, yedges_b, hist_b = axs[0, 1].hist2d(x_b, y_b, bins=(xedges, yedges), range=bin_range, cmap=cmap, density=density, vmin=vmin_hist,vmax=vmax_hist)
    axs[0, 1].set_title('10b')

    # compute the difference between the nominal and target distributions
    difference = h_a - h_b
    max_abs = np.abs(difference).max()
    difference_fig = axs[0, 2].pcolormesh(xedges, yedges, difference.T, cmap='bwr', vmin=-max_abs, vmax=max_abs)
    cbar = fig.colorbar(difference_fig, ax=axs[0, 2])
    axs[0, 2].set_title('10a - 10b')

    # Reweighted distribution
    reweighted, reweighted_edges, reweighted_bins, reweighted_image = axs[1, 1].hist2d(
        x_a, 
        y_a, 
        bins=bins, 
        range=bin_range, 
        cmap=cmap, 
        weights=weights, 
        density=density, 
        vmin=vmin_hist,
        vmax=vmax_hist
    );
    axs[1, 1].set_title('a reweighted')

    # Compute difference between reweighted and target
    difference_reweighted = (reweighted - h_b)
    difference_reweighted_fig = axs[1, 2].pcolormesh(xedges, yedges, difference_reweighted.T, cmap='bwr', vmin=difference_fig.norm.vmin, vmax=difference_fig.norm.vmax)
    cbar = fig.colorbar(difference_reweighted_fig, ax=axs[1, 2])
    axs[1, 2].set_title('reweighted - b')

    for ax in axs.flat:
        ax.set(xlabel=xlabel, ylabel=ylabel)
    


def plot_distribution(nominal,
                      target,
                      weights,
                      weights_target=None,
                      label_nominal=None,
                      label_target=None,
                      n_bins=None,
                      range=None,
                      errorbars=False,
                      density=True,
                      ratio_limits=None,
                      kl_div=True):
    plt.rcParams['figure.figsize'] = (12, 10)
    fig, axs = plt.subplots(2, 1)

    # Create legend labels
    if label_nominal is None:
        label_nominal = 'Nominal'
    if label_target is None:
        label_target = 'Target'
    label_reweighted = f'{label_nominal} to {label_target}'

    if weights_target is None:
        weights_target = np.ones_like(target)
    # Histograms
    counts_ref, bins_ref, _ = axs[0].hist(target,
                                          weights=weights_target,
                                          label=label_target,
                                          bins=n_bins,
                                          **plot_style_0,
                                          range=range,
                                          density=density)
    counts_nominal, _, _ = axs[0].hist(nominal,
                                       label=label_nominal,
                                       bins=n_bins,
                                       **plot_style_1,
                                       range=range,
                                       density=density)
    counts_reweighted, _, _ = axs[0].hist(nominal,
                                          weights=weights,
                                          bins=n_bins,
                                          label=label_reweighted,
                                          **plot_style_1,
                                          range=range,
                                          density=density)
    axs[0].legend()

    # Ratio Plots
    ratio_plot(bins_ref,
               counts_ref,
               counts_nominal,
               axs,
               errorbars,
               color='blue',
               label=label_nominal,
               ratio_limits=ratio_limits)
    ratio_plot(bins_ref,
               counts_ref,
               counts_reweighted,
               axs,
               errorbars,
               color='orange',
               label=label_reweighted,
               ratio_limits=ratio_limits)

    # Add KL vals to ratio plots
    if kl_div:
        kl_nominal = kl_divergence(counts_ref, counts_nominal)
        kl_reweighted = kl_divergence(counts_ref, counts_reweighted)
        
        # Build a rectangle in axes coords
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        axs[1].text(
            right, 
            top + .15, 
            f'KL Nominal: {kl_nominal:.2f}\n KL Reweighted: {kl_reweighted:.2f}',
            horizontalalignment='right',
            verticalalignment='top',
            size='xx-small',
            transform=axs[1].transAxes)


def ratio_plot(bins_ref,
               counts_ref,
               counts,
               axs,
               errorbars,
               color,
               label,
               ratio_limits=None):
    if ratio_limits is None:
        ratio_limits = (0.8, 1.2)
    ratio_limit_low, ratio_limit_high = ratio_limits
    x_values, y_values, y_errors = ratios(bins_ref, counts_ref, counts)
    axs[1].plot(x_values, np.ones_like(counts_ref), '--')
    if errorbars:
        axs[1].fill_between(x_values,
                            y_values - y_errors,
                            y_values + y_errors,
                            alpha=0.5)

    axs[1].plot(x_values, y_values, 'o', color=color, label=label)
    axs[1].set_ylim(ratio_limit_low, ratio_limit_high)
    axs[1].set_title('Ratio to Target')


def ratios(bins_ref, counts_ref, counts):
    x_values = [
        0.5 * (bins_ref[i + 1] + bins_ref[i]) for i in range(len(counts_ref))
    ]

    y_values = (counts + 1e-9) / (counts_ref + 1e-9)
    y_errors = np.sqrt(counts)/(counts + 1e-9) \
             + np.sqrt(counts_ref)/(counts_ref + 1e-9)
    y_errors *= y_values

    return x_values, y_values, y_errors
