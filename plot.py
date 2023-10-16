import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from utils.funcs import load_files
from utils.plotting import plot_distribution
from functools import partial
from utils.funcs import load_config
from config.config import ReweightVariable
from config.config import ReweightConfig
from typing import List
import pandas as pd


def make_plots(nominal, target, weights, config: ReweightConfig):
    # save all the plots in a big pdf
    with PdfPages(f'{plots_path}/all_plots.pdf') as pdf:
        for variable in config.reweight_variables:
            _plot(nominal, target, variable, weights)
            pdf.savefig()



def _plot(nominal: pd.DataFrame,
          target: pd.DataFrame,
          reweight_variable: ReweightVariable,
          weights: np.array,
          custom_name=''):
    plot_distribution(label_nominal=config.nominal_name,
                      label_target=config.target_name,
                      nominal=nominal[reweight_variable.name],
                      target=target[reweight_variable.name],
                      n_bins=reweight_variable.n_bins,
                      range=reweight_variable.hist_range,
                      weights=weights)
    suptitle = f'{reweight_variable.name} {custom_name}'
    plt.suptitle(suptitle)
    # check if the directory exists create it if not
    if not os.path.exists(f'{plots_path}/png_format'):
        os.makedirs(f'{plots_path}/png_format')

    plt.savefig(
        f"{plots_path}/png_format/{reweight_variable.name}{custom_name}.png")


def plots_at_bins(nominal: pd.DataFrame,
                  target: pd.DataFrame,
                  predicted_weights: np.array,
                  config: ReweightConfig,
                  bin_size=1,
                  num_bins=5,):
    
    absolute_bias = config.get_variable_by_name('Erec_bias_abs')
    relative_bias = config.get_variable_by_name('Erec_bias_rel')
    bins = np.arange(num_bins, step=bin_size)

    with PdfPages(f'{plots_path}/true_energy_slices.pdf') as pdf:
        for idx in range(num_bins - 1):
            current_bin = bins[idx]
            next_bin = bins[idx + 1]
            true_energy = nominal['Enu_true']
            true_energy_target = target['Enu_true']
            mask_nominal = (true_energy > current_bin) & (true_energy < next_bin)
            
            current_nominal = nominal[mask_nominal]
            current_weights = predicted_weights[mask_nominal]
            current_target = target[(true_energy_target > current_bin) & (true_energy_target < next_bin)]
            
            current_name = f'[{current_bin}, {next_bin}]_GeV'
            _plot(current_nominal, current_target, absolute_bias, current_weights, current_name)
            pdf.savefig()
            _plot(current_nominal, current_target, relative_bias, current_weights, current_name)
            pdf.savefig()
    return None


config = load_config(path='config/hA2018_to_noFSI.yaml')

#Load weights
weights = np.load(
    f'trained_bdt/{config.nominal_name}_to_{config.target_name}/weights.npy')

plots_path = os.path.join(config.plots_path,
                          f'{config.nominal_name}_to_{config.target_name}')
# Plot weights

plt.hist(weights, bins=100)
plt.yscale('log')
plt.title('Weights')
if not os.path.exists(f'{plots_path}'):
    os.makedirs(f'{plots_path}')
plt.savefig(f"{plots_path}/hist.png")
plt.show()

nominal = load_files(config.nominal_files,
                     config.reweight_variables_names,
                     return_dataframe=True)[:config.number_of_train_events]
target = load_files(config.target_files,
                    config.reweight_variables_names,
                    return_dataframe=True)[:config.number_of_train_events]
# Check that files were loaded
assert len(nominal) > 0, 'No files found for nominal'
plots_at_bins(nominal, target, weights, config)
make_plots(nominal, target, weights, config)
