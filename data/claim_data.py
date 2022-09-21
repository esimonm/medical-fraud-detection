import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold

def process_mil_claim_data(path_X, path_Y, k, load_from_pickle=False, pickle_path=''):
    """
    Process the medical claims data for MIL from pre-processed csv files.
    Parameters
    ---------------------
    path_X : string
        Path to the features .csv file.
    path_Y : string
        Path to the labels .csv file.
    k: int
        Number of Folds.
    load_from_pickle: bool
        Indicates whether to load the dataset from pickle.
    pickle_path: string
        Path of pickle file to import.
    Return
    ---------------------
    dataset : 
        .
    """

    # Import data as numpy arrays
    print('Importing dataset ...')

    if load_from_pickle and pickle_path:
        datasets = pickle.load(open(pickle_path, "rb"))
        return datasets

    x = pd.read_csv(path_X).to_numpy()
    x = np.concatenate((x, np.expand_dims(np.arange(x.shape[0]),1)), axis=1)

    y = pd.read_csv(path_Y).to_numpy()

    # Get array of unique providers
    providers_data = pd.DataFrame(y[:,1:]).drop_duplicates().to_numpy()

    # Create bags
    bag_count = providers_data[:,0].shape[0]
    print('Creating {0:6d} bags ...'.format(bag_count))

    bags_fea = []

    provider_codes = list(y[:,1])
    instance_indeces = list(y[:,0])

    for i, provider in enumerate(providers_data[:,0]):
        print('Creating Bag of Provider {0:6d}/{1:6d}'.format(i, bag_count))
        instance_indeces = [i for i, p in enumerate(provider_codes) if p == provider]
        matrix = x[instance_indeces, 1:]
        vector = y[instance_indeces, 2]
        bags_fea.append((matrix, list(vector)))
    
    bag_idxs = np.arange(len(bags_fea))
    bag_cut = int(np.floor(len(bags_fea) * 0.80))

    train_idxs = bag_idxs[:bag_cut]
    test_idxs = bag_idxs[bag_cut:]

    bags_fea_train = [bags_fea[ibag] for ibag in train_idxs]
    bags_fea_test = [bags_fea[ibag] for ibag in test_idxs]

    # KFold - split train into train and validation
    print('Creating K-folds ...')

    datasets = []

    if k == 1:
        idxs = np.arange(len(bags_fea_train)) 
        np.random.shuffle(idxs)
        
        cut = int(np.floor(len(bags_fea_train) * 0.80))

        train_idxs = idxs[:cut]
        test_idxs = idxs[cut:]
        
        dataset = {}
        dataset['train'] = [bags_fea_train[ibag] for ibag in train_idxs]
        # validation. we call it test because then it is referenced as test
        dataset['test'] = [bags_fea_train[ibag] for ibag in test_idxs]
        datasets.append(dataset)
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=None)
        for train_idx, test_idx in kf.split(bags_fea_train):
            dataset = {}
            dataset['train'] = [bags_fea_train[ibag] for ibag in train_idx]
            # validation. we call it test because then it is referenced as test
            dataset['test'] = [bags_fea_train[ibag] for ibag in test_idx]
            datasets.append(dataset)

    # Convert test into needed format
    dataset_test = {
        'test': bags_fea_test
    }

    print('Data imported succesfully!')

    # Save
    pickle.dump(datasets, open("data/datasets_train_val.pkl", "wb"))
    pickle.dump(dataset_test, open("data/dataset_test.pkl", "wb"))

    print('Data pickled succesfully!')

    return datasets

def load_mil_pickled_data(data_path):
    """
    Load pre-processed data from pickle for MIL.
    """
    datasets = pickle.load(open(data_path, "rb"))

    X, y = [], []

    print("Preparing train data...")
    for i, tup in enumerate(datasets[0]['train']):
        X.append(tup[0])
        y.append(tup[1])

    print("Train data ready!")

    X_valid, y_valid = [], []

    print("Preparing validation data...")
    for i, tup in enumerate(datasets[0]['test']):
        X_valid.append(tup[0])
        y_valid.append(tup[1])

    print("Validation data ready!")

    del datasets

    shape = X[0].shape
    print('First train datapoint shape: ', shape)

    return X, y, X_valid, y_valid
