import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.utils.funcs import get_vars_meta, rootfile_to_array
from src.utils.plotting import plot_distribution
from trained_bdt.genie_v3_10b import GENIEv3_10b_BDT

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_dir", type=str, default="/eos/home-c/cristova/DUNE/AlternateGenerators/"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None)
args = parser.parse_args()


def make_plots(nominal, other, predicted_weights, vars_meta, figsize=None):
    if figsize is not None:
        plt.rcParams["figure.figsize"] = figsize

    # Create variable names    
    col_names, n_bins, x_min, x_max, fig_title = vars_meta

    # Iterate through variables and plot
    for idx, variable in enumerate(col_names):
        hist_range = (float(x_min[idx]), float(x_max[idx]))
        plot_distribution(
            nominal[:, idx],
            other[:, idx],
            predicted_weights,
            n_bins=int(n_bins[idx]),
            errorbars=False,
            density=True,
            range=hist_range,
            label_nominal="GENIEv2",
            label_target="GENIEv3",
            ratio_limits=(0.8, 1.2),
        )
        plt.suptitle(variable)
        if 'is' not in variable:
            plt.savefig(f"saved_plots/{variable}.png")
        else:
            plt.show()


def calculate_weights(probas):
    weights = np.clip(probas, 0, 1000)
    weights = np.exp(weights)
    weights = np.squeeze(np.nan_to_num(weights))
    return weights


generator_a = 'flat_argon_12*_GENIEv2'
generator_b = 'flat_argon_12*_GENIEv3_G18_10b'


model = GENIEv3_10b_BDT()


v2_filename = args.root_dir + f'{generator_a}_1M_04[0-9]_NUISFLAT.root'
v3_filename = args.root_dir + f'{generator_b}*1M_04[0-9]_NUISFLAT.root'

def predict_weights(filename):
    filenames = glob.glob(filename)
    batch_size = 1000
    weights_list = []
    probas_list = []
    nominal_list = []

    for filename in filenames:
        nominal = rootfile_to_array(filename)
        n_iterations = int(len(nominal)/batch_size)
        for i in tqdm(range(n_iterations)):
            
            # Indices to low through model
            idx_low = i * batch_size
            idx_high = idx_low + batch_size

            # Get predictions
            y_hat = model.predict(nominal[idx_low:idx_high])
            # Get weights      
            weights = calculate_weights(y_hat)
            weights_list.append(weights)
            probas_list.append(y_hat)
        
        # Save the nominal file
        nominal_list.append(nominal)
        
        # After iterating through create weight files
        weights = np.hstack(weights_list)
        probas = np.vstack(probas_list)
        nominal = np.vstack(nominal_list)
    return weights, probas, nominal

# Get weights for nominal array
weights, probas_nominal, nominal = predict_weights(v2_filename)

# Get target array
target = np.vstack([rootfile_to_array(filename) for filename in glob.glob(v3_filename)])

plt.hist(weights, bins=100)
plt.yscale('log')
plt.savefig(f"saved_plots/hist.png")

figsize = (12, 10)
many_bins = 100
vars_meta = get_vars_meta(many_bins)
make_plots(nominal, target, weights, vars_meta, figsize)




