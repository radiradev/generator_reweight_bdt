import os 
import configparser
import ast
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from utils.funcs import load_files
from utils.plotting import plot_distribution
from config.plots import plots_meta
from functools import partial


# Parse config
config = configparser.ConfigParser()
config.read('config/files.ini')
test = config['test']
generator_a = test['generator_a']
generator_b = test['generator_b']
nominal_filenames = ast.literal_eval(test['filepaths_a'])
target_filenames = ast.literal_eval(test['filepaths_b'])

plots_path = f'plots/{generator_a}_vs_{generator_b}'
# check if the directory exists
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

def make_plots(nominal, other, predicted_weights, target_weights, plots_meta, figsize=None, plots_path='saved_plots'):
    """
    Make plots for all variables in plots_meta.

    Args:
        nominal (np.array): array of nominal values
        other (np.array): array of other values
        predicted_weights (np.array): array of predicted weights
        target_weights (np.array): array of target weights
        plots_meta (dict): dictionary of variables to plot
        figsize (tuple): figure size
        plots_path (str): path to save plots
    
    Returns:
        None
    """
    if figsize is not None:
        plt.rcParams["figure.figsize"] = figsize

    # save all the plots in a big pdf
    with PdfPages(f'{plots_path}/all_plots.pdf') as pdf:
        # invert columns for plotting
        nominal = np.transpose(nominal)
        other = np.transpose(other) 

        col_names = plots_meta.keys()

        plot_current_distribution = partial(
            plot_distribution,
            weights=predicted_weights,
            weights_target=target_weights,
            errorbars=False,
            density=True,
            label_nominal=generator_a,
            label_target=generator_b,
            ratio_limits=(0.8, 1.2),)


        for idx, col_title in enumerate(col_names):
            variable = plots_meta[col_title]
            plot_current_distribution(
                nominal[idx],
                other[idx],
                n_bins=variable.n_bins,
                range=variable.hist_range,
            )
            if variable.train:
                suptitle = col_title + ' (train)'
            else:
                suptitle = col_title + ' (test)'
            plt.suptitle(suptitle)
            
            pdf.savefig()

            # check if the directory exists create it if not
            if not os.path.exists(f'{plots_path}/png_format'):
                os.makedirs(f'{plots_path}/png_format')
            
            plt.savefig(f"{plots_path}/png_format/{col_title}.png")


#Load weights
weights = np.load(f'trained_bdt/{generator_a}_to_{generator_b}/weights.npy') 


# Plot weights
plt.hist(weights, bins=100)
plt.yscale('log')
plt.title('Weights')
plt.savefig(f"{plots_path}/hist.png")
plt.show()


nominal = load_files(nominal_filenames, test_data=True)
target, target_weights = load_files(target_filenames, test_data=True, return_weights=True)

# Check that files were loaded
assert len(nominal) > 0, 'No files found for nominal'

figsize = (12, 10)
n_bins = 100
vars_meta = plots_meta(n_bins)
make_plots(nominal, target, weights, target_weights, vars_meta, figsize, plots_path=plots_path)
