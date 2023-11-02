import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from utils.funcs import load_files
from utils.plotting import plot_distribution
from utils.funcs import load_config
from config.config import ReweightVariable
from config.config import ReweightConfig
import pandas as pd
import fire

def make_plots(nominal, target, weights, config: ReweightConfig, plot_name: str):
    # save all the plots in a big pdf
        
    with PdfPages(f'{plots_path}/{plot_name}_reweight_variables.pdf') as pdf:
        for variable in config.reweight_variables:
            _plot(nominal, target, variable, weights, plot_name)
            pdf.savefig()

def _plot(
          nominal: pd.DataFrame,
          target: pd.DataFrame,
          reweight_variable: ReweightVariable,
          weights: np.array,
          plot_name: str,
          custom_name='',
          ):
    plot_distribution(label_nominal=config.nominal_name,
                      label_target=config.target_name,
                      nominal=nominal[reweight_variable.name],
                      target=target[reweight_variable.name],
                      n_bins=reweight_variable.n_bins,
                      range=reweight_variable.hist_range,
                      weights=weights)
    suptitle = f'{reweight_variable.name} {custom_name}'
    plt.suptitle(suptitle)

    plt.savefig(
        f"{plots_path}/png_format_{plot_name}/{reweight_variable.name}{custom_name}.png")

def check_plots_path(config: ReweightConfig):
    plots_path = os.path.join(config.plots_path,
                          f'{config.nominal_name}_to_{config.target_name}')
    if not os.path.exists(f'{plots_path}'):
        os.makedirs(f'{plots_path}')
        os.makedirs(os.path.join(plots_path, 'png_format_near')) 
        os.makedirs(os.path.join(plots_path, 'png_format_far_oscillated'))
    return plots_path

def plot_weights(weights, name):
    plt.hist(weights, bins=100)
    plt.yscale('log')
    plt.title('Weights')
    plt.savefig(f"{plots_path}/{name}.png")
    plt.show()
    

def process_bin(idx, bins, weights, nominal, target):
    current_bin = bins[idx]
    next_bin = bins[idx + 1]
    true_energy = nominal['Enu_true']
    true_energy_target = target['Enu_true']
    mask_nominal = (true_energy > current_bin) & (true_energy < next_bin)
    
    current_nominal = nominal[mask_nominal]
    current_weights = weights[mask_nominal]
    current_target = target[(true_energy_target > current_bin) & (true_energy_target < next_bin)]
    
    current_name = f'[{current_bin}, {next_bin}]_GeV'
    return current_nominal, current_target, current_weights, current_name

    
def plots_at_bins(nominal: pd.DataFrame,
                  target: pd.DataFrame,
                  predicted_weights: np.array,
                  config: ReweightConfig,
                  plot_name: str,
                  bin_size=1,
                  num_bins=5,
                  ):
    
    absolute_bias = config.get_variable_by_name('Erec_bias_abs')
    relative_bias = config.get_variable_by_name('Erec_bias_rel')
    bins = np.arange(num_bins, step=bin_size)
    print(plot_name)
    with PdfPages(f'{plots_path}/{plot_name}_true_energy_slices.pdf') as pdf:
        for bias_type in [absolute_bias, relative_bias]:
            for idx in range(num_bins - 1):
                current_nominal, current_target, current_weights, current_name = process_bin(idx, bins, predicted_weights, nominal, target)
                _plot(current_nominal, current_target, bias_type, current_weights, plot_name, current_name)
                pdf.savefig()
    return None



def main(config_name):
    global config
    config = load_config(path=f'config/{config_name}')
    #Load weights
    weights = np.load(
        f'trained_bdt/{config.nominal_name}_to_{config.target_name}/weights.npy')
    
    global plots_path
    plots_path = check_plots_path(config)

    # Plot weights
    plot_weights(weights, name='near')

    nominal = load_files(config.plotting_nominal,
                        config.reweight_variables_names)
    target = load_files(config.plotting_target,
                        config.reweight_variables_names)
    if config.number_of_train_events != -1:
        # TODO need to refactor 
        nominal = nominal[:config.number_of_train_events]
        target = target[:config.number_of_train_events]

    # Check that files were loaded
    assert len(nominal) > 0, 'No files found for nominal'

    plots_at_bins(nominal, target, weights, config, plot_name='near')
    make_plots(nominal, target, weights, config, plot_name='near')

    # plot oscillated files


    weights_oscillated = np.load(
        f'trained_bdt/{config.nominal_name}_to_{config.target_name}/weights_oscillated.npy')
    assert np.array_equal(weights, weights_oscillated) == False, "Should have different shapes"
    plot_weights(weights_oscillated, name='far_oscillated')

    nominal_oscillated = load_files(
        config.plotting_nominal_oscillated,
        config.reweight_variables_names,
    )
    target_oscillated = load_files(
        config.plotting_target_oscillated,
        config.reweight_variables_names
    )
    plots_at_bins(nominal_oscillated, target_oscillated, weights_oscillated, config, plot_name='far_oscillated')
    make_plots(nominal_oscillated, target_oscillated, weights_oscillated, config, plot_name='far_oscillated')

if __name__ == '__main__':
    fire.Fire(main)