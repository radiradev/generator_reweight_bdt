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
    if range is None:
        min_val = 0
        max_val = np.min([np.max(nominal), np.max(target)])
        range = (min_val, max_val)

    fig, axs = plt.subplots(2, 1)

    # Create legend labels
    if label_nominal is None:
        label_nominal = 'Nominal'
    if label_target is None:
        label_target = 'Target'
    label_reweighted = f'{label_nominal} to {label_target}'

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
        ratio_limits = (0.5, 1.5)
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
