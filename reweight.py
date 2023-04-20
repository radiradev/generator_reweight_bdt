import numpy as np
import joblib
import configparser
import ast

from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils.funcs import rootfile_to_array, sigmoid
from utils.funcs import get_variables_out

# Parse config
config = configparser.ConfigParser()
config.read('config/files.ini')
test = config['test']
generator_a = test['generator_a']
generator_b = test['generator_b']
data_dir = test['data_dir']
nominal_filenames = ast.literal_eval(test['filepaths_a'])

checkpoint_path = f'trained_bdt/{generator_a}_to_{generator_b}/BDT.pkl'

def predict_weights(filenames, model):
    """
    Predict weights for a list of nominal filenames using a trained model

    Args:
        filenames (list): List of filenames to predict weights for
        model (sklearn model): Trained model to use for prediction

    Returns:
        weights (np.array): Array of weights
        probas (np.array): Array of probabilities
        nominal (np.array): Array of nominal values
    """
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
            batch = nominal[idx_low:idx_high]
            y_hat = model.predict_proba(batch)[:, 1]
            weights = model.predict_proba(batch)[:, 1]/model.predict_proba(batch)[:, 0]
            
            # Get weights     
            weights_list.append(weights)
            probas_list.append(sigmoid(y_hat))

        # Save the nominal file
        nominal_list.append(nominal)
            
    # After iterating through create weight files
    weights = np.hstack(weights_list)
    probas = np.hstack(probas_list)
    nominal = np.vstack(nominal_list)
    return weights, probas, nominal



# Load sklearn model
model = joblib.load(checkpoint_path)
# Get weights for nominal array
weights, probas_nominal, nominal = predict_weights(nominal_filenames, model)

np.save(f'trained_bdt/{generator_a}_to_{generator_b}/weights.npy', weights)

