import numpy as np
import joblib
import warnings
from tqdm import tqdm
from utils.funcs import rootfile_to_array, load_config


def predict_weights(filenames, model, num_of_train_events, batch_size=None):
    """
    Predict weights for a list of nominal filenames using a trained model

    Args:
        filenames (list): List of filenames to predict weights for
        model (sklearn model): Trained model to use for prediction

    Returns:
        weights (np.array): Array of weights
    """
    num_files = len(filenames) if isinstance(filenames, list) else 1
    file_length = num_of_train_events # test data is half of train data
    if batch_size is None:
        warnings.warn("Currently batch_size not None only works for a single file")
        nominal = rootfile_to_array(filenames, config.reweight_variables_names)[:config.number_of_train_events]
        probas = model.predict_proba(nominal)
        weights = probas[:, 1]/probas[:, 0]

    else:
        warnings.warn("This method is outdated. Refactor before using again.")
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
        
        weights_array = weights_array.flatten()

            
    return weights 


config = load_config(path='config/hA2018_to_noFSI.yaml')
ckpt_path = f'trained_bdt/{config.nominal_name}_to_{config.target_name}/BDT.pkl'
# Load sklearn model
model = joblib.load(ckpt_path)
# Get weights for nominal array
weights = predict_weights(config.nominal_files, model, config.number_of_train_events)

np.save(f'trained_bdt/{config.nominal_name}_to_{config.target_name}/weights.npy', weights)