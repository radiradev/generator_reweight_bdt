import numpy as np
import joblib
import warnings
from tqdm import tqdm
from utils.funcs import rootfile_to_array, load_config, load_files
from functools import partial
import pandas as pd 
import fire


def predict_weights(config, filenames, model):
    """
    Predict weights for a list of nominal filenames using a trained model

    Args:
        filenames (list): List of filenames to predict weights for
        model (sklearn model): Trained model to use for prediction

    Returns:
        weights (np.array): Array of weights
    """

    nominal = load_files(filenames, config.reweight_variables_names)
    probas = model.predict_proba(nominal)
    weights = probas[:, 1]/probas[:, 0]
    return weights


def reweight(config_name):
    config = load_config(path=f"config/{config_name}")
    ckpt_path = f'trained_bdt/{config.nominal_name}_to_{config.target_name}/BDT.pkl'
    # Load sklearn model
    model = joblib.load(ckpt_path)
    # Get weights for nominal array
    weights = predict_weights(config, config.plotting_nominal, model)
    np.save(f'trained_bdt/{config.nominal_name}_to_{config.target_name}/weights.npy', weights)

    # TODO check if this exists in the config
    oscillated_weights = predict_weights(config, config.plotting_nominal_oscillated, model)
    np.save(f'trained_bdt/{config.nominal_name}_to_{config.target_name}/weights_oscillated.npy', oscillated_weights)

if __name__ == '__main__':
    fire.Fire(reweight)