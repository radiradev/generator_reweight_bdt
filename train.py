import pickle
import os
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils.funcs import load_files
from utils.funcs import load_config
import pandas as pd
import fire


def train_classifier(data, labels, filename, weights=None, max_iter=500, max_depth=15):
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
    classifier = LGBMClassifier(boosting_type='gbdt', verbose=1)

    # TODO investigate nan values
    data_train = np.nan_to_num(data_train)
    data_test = np.nan_to_num(data_test)
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




def load_data(config, num_events=None):
    # Load data
    nominal = load_files(config.nominal_files, config.reweight_variables_names)
    target = load_files(config.target_files, config.reweight_variables_names)
    if num_events != -1:
        nominal = nominal[num_events:]
        target = target[num_events:]
    # Concatenate datas
    data = pd.concat([nominal, target], ignore_index=True)
    print(f'Data shape {data.shape}')
    labels = np.concatenate((np.zeros(len(nominal)), np.ones(len(target))))
    return data, labels

# load data
def train(config_name):
    config = load_config(path=os.path.realpath(f"config/{config_name}"))
    data, labels = load_data(config, config.number_of_train_events, )
    print(data.head())
    # train the bdt
    ckpt_path = f'trained_bdt/{config.nominal_name}_to_{config.target_name}/BDT.pkl'
    # check if the directory exists and create it if not
    if not os.path.exists(os.path.dirname(ckpt_path)):
        os.makedirs(os.path.dirname(ckpt_path))

    train_classifier(data, labels, ckpt_path)

if __name__ == "__main__":
    fire.Fire(train)

