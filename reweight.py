import numpy as np
import joblib
import configparser
import ast

from tqdm import tqdm
from utils.funcs import rootfile_to_array, sigmoid


# Parse config
config = configparser.ConfigParser()
config.read('config/files.ini')
test = config['test']
generator_a = test['generator_a']
generator_b = test['generator_b']
data_dir = test['data_dir']
nominal_filenames = ast.literal_eval(test['filepaths_a'])

checkpoint_path = f'trained_bdt/{generator_a}_to_{generator_b}/BDT.pkl'

def predict_weights(filenames, model, batch_size=1000, file_length=1_000_000):
    """
    Predict weights for a list of nominal filenames using a trained model

    Args:
        filenames (list): List of filenames to predict weights for
        model (sklearn model): Trained model to use for prediction

    Returns:
        weights (np.array): Array of weights
    """
    num_files = len(filenames)
    weights_array = np.zeros(shape=(file_length, num_files))
    print(weights_array.shape)
    for j, filename in enumerate(filenames):
        nominal = rootfile_to_array(filename)
        assert len(nominal) == file_length, f"File {filename} does not have specified length {file_length}"
        n_iterations = int(len(nominal)/batch_size)
        for i in tqdm(range(n_iterations)):

            # Indices to low through model
            idx_low = i * batch_size
            idx_high = idx_low + batch_size

            # Get predictions
            batch = nominal[idx_low:idx_high]
            batch = np.nan_to_num(batch)
            probas = model.predict_proba(batch)
            weights = probas[:, 1]/probas[:, 0]
            
            # Get weights     
            weights_array[idx_low:idx_high, j] = weights

            
    return weights_array.flatten()



# Load sklearn model
model = joblib.load(checkpoint_path)
# Get weights for nominal array
weights = predict_weights(nominal_filenames, model)

np.save(f'trained_bdt/{generator_a}_to_{generator_b}/weights.npy', weights)

