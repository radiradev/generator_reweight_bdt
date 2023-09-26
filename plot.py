import os 
import configparser
import ast
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from utils.funcs import load_files
from utils.plotting import plot_distribution
from functools import partial
from utils.funcs import load_config
from config.config import ReweightVariable
from typing import List


def make_plots(nominal, other, predicted_weights, target_weights, reweight_variables: List[ReweightVariable], figsize=None, plots_path='saved_plots'):
    """
    Make plots for all variables in plots_meta.

    Args:
        nominal (np.array): array of nominal values
        other (np.array): array of other values
        predicted_weights (np.array): array of predicted weights
        target_weights (np.array): array of target weights
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

        plot_current_distribution = partial(
            plot_distribution,
            weights=predicted_weights,
            weights_target=target_weights,
            errorbars=False,
            density=True,
            label_nominal=config.nominal_name,
            label_target=config.target_name,
            ratio_limits=(0.8, 1.2),)


        for idx, reweight_variable in enumerate(reweight_variables):
            plot_current_distribution(
                nominal[idx],
                other[idx],
                n_bins=reweight_variable.n_bins,    
                range=reweight_variable.hist_range,
            )
            if not reweight_variable.observer:
                suptitle = reweight_variable.name + ' (reweighting variable)'
            else:
                suptitle = reweight_variable.name + ' (observer variable)'
            plt.suptitle(suptitle)
            
            pdf.savefig()

            # check if the directory exists create it if not
            if not os.path.exists(f'{plots_path}/png_format'):
                os.makedirs(f'{plots_path}/png_format')
            
            plt.savefig(f"{plots_path}/png_format/{reweight_variable.name}.png")


config = load_config(path='config/hA2018_to_noFSI.yaml')

#Load weights
weights = np.load(f'trained_bdt/{config.nominal_name}_to_{config.target_name}/weights.npy')


plots_path = os.path.join(config.plots_path, f'{config.nominal_name}_to_{config.target_name}')
# Plot weights

print(weights.shape)
plt.hist(weights, bins=100)
plt.yscale('log')
plt.title('Weights')
if not os.path.exists(f'{plots_path}'):
    os.makedirs(f'{plots_path}')
plt.savefig(f"{plots_path}/hist.png")
plt.show()


nominal = load_files([config.nominal_files], variables_out=config.reweight_variables_names)[:config.number_of_train_events]
target = load_files([config.target_files], variables_out=config.reweight_variables_names)[:config.number_of_train_events]
target_weights = np.ones(len(target))
# Check that files were loaded
assert len(nominal) > 0, 'No files found for nominal'

figsize = (12, 10)
make_plots(nominal, target, weights, target_weights, config.reweight_variables, figsize, plots_path=plots_path)
