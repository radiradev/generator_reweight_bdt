import pickle
import os
import configparser
import ast
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils.funcs import load_files


# Parse config
config = configparser.ConfigParser()
config.read('config/files.ini')
train = config['train']
generator_a = train['generator_a']
generator_b = train['generator_b']
nominal_filename = ast.literal_eval(train['filepaths_a'])
target_filename = ast.literal_eval(train['filepaths_b'])


def train_classifier(data, labels, filename, weights=None, max_iter=100, max_depth=5):
    """
    Train a classifier and save it to a pickle file

    Args:
        data (np.array): Array of data
        labels (np.array): Array of labels
        filename (str): Name of file to save classifier to
        weights (np.array): Optional array of sample weights
        max_iter (int): Number of iterations to train classifier for
        max_depth (int): Max depth of classifier
    """

    # Split data into train and test
    if weights is None:
        data_train, data_test, labels_train, labels_test = train_test_split(
            data, labels, test_size = 0.2)
        # free up memory
        del data
        del labels
    else:
        data_train, data_test, labels_train, labels_test, weights_train, weights_test = train_test_split(
            data, labels, weights, test_size = 0.2)

    #Fit reweighter
    classifier = HistGradientBoostingClassifier(max_iter=max_iter, verbose=1, max_depth=max_depth, l2_regularization=0.1)
    if weights is None:
        classifier.fit(data_train, labels_train)
    else:
        classifier.fit(data_train, labels_train, sample_weight=weights_train)

    # save the bdt in a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(classifier, f)
    
    if weights is None:
        clf_score = roc_auc_score(labels_test, classifier.predict_proba(data_test)[:,1])
    else:
        clf_score = roc_auc_score(labels_test, classifier.predict_proba(data_test)[:,1], sample_weight=weights_test)
    print(f'ROC score: {clf_score}')



# Load data
nominal = load_files(nominal_filename, return_weights=False)
target = load_files(target_filename, return_weights=False)

# Concatenate data
data = np.concatenate((nominal, target))
print(f'Data shape {data.shape}')
labels = np.concatenate((np.zeros(len(nominal)), np.ones(len(target))))

# train the bdt
save_path = f'trained_bdt/{generator_a}_to_{generator_b}/BDT.pkl'
# check if the directory exists and create it if not
if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))

train_classifier(data, labels, save_path)


