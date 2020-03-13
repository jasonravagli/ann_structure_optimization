from enum import Enum
import keras
import numpy as np
from scipy.io import arff
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

class DatasetPreprocessing(Enum):
    NO_PREPROCESSING = 0
    STANDARDIZE = 1
    NORMALIZE = 2

def read_diabetic_retinopathy_debrecen(n_fold, generate_validation_set, preprocessing_mode=DatasetPreprocessing.NO_PREPROCESSING):
    """
    Reads the Diabetic Retinopathy Debrecen Dataset (binary classification) from the messidr_featuresd.arff file
    :return: Dataset features and labels as three separate numpy arrays for each type of set (train, validation and test)
    """

    data, meta = arff.loadarff("datasets/messidor_features.arff")

    features_cols = meta.names()[:-1]

    # Features are read as array of tuples: they need to be converted into array of arrays
    features = data[features_cols]
    features = np.asarray(features.tolist())
    labels = data["Class"]

    # Feature selection
    new_features = []
    new_labels = []
    for i in range(len(features)):
        if features[i, 0] == 1:
            new_features.append(features[i, 1:])
            new_labels.append(labels[i])

    features = np.array(new_features)
    labels = np.array(new_labels)

    # Strtatified k-fold
    skf = StratifiedKFold(n_splits=n_fold)
    train_indexes = []
    valid_and_test_indexes = []
    for train, valid_and_test in skf.split(features, labels):
        train_indexes = train
        valid_and_test_indexes = valid_and_test
        break

    x_train = features[train_indexes]
    if generate_validation_set:
        y_train = keras.utils.to_categorical(labels[train_indexes], 2)
    else:
        y_train = labels[train_indexes]
    x_valid_and_test = features[valid_and_test_indexes]
    y_valid_and_test = labels[valid_and_test_indexes]

    # Data preprocessing
    if preprocessing_mode == DatasetPreprocessing.STANDARDIZE:
        x_train = preprocessing.scale(x_train)
        x_valid_and_test = preprocessing.scale(x_valid_and_test)
    elif preprocessing_mode == DatasetPreprocessing.NORMALIZE:
        x_train = preprocessing.normalize(x_train)
        x_valid_and_test = preprocessing.normalize(x_valid_and_test)


    if generate_validation_set:
        # Split equally validation and test set
        skf = StratifiedKFold(n_splits=2)
        valid_indexes = []
        test_indexes = []
        for valid, test in skf.split(x_valid_and_test, y_valid_and_test):
            valid_indexes = valid
            test_indexes = test
            break

        x_valid = x_valid_and_test[valid_indexes]
        y_valid = keras.utils.to_categorical(y_valid_and_test[valid_indexes], 2)
        x_test = x_valid_and_test[test_indexes]
        y_test = keras.utils.to_categorical(y_valid_and_test[test_indexes], 2)
    else:
        x_valid = None
        y_valid = None
        x_test = x_valid_and_test
        y_test = keras.utils.to_categorical(y_valid_and_test, 2)

    return x_train, y_train, x_valid, y_valid, x_test, y_test
