'''A script for preprocessing the dataset into the correct form
'''

import pandas as pd
from feature_engineering import *

# ignore pandas '+' operator warning
import warnings
warnings.filterwarnings('ignore')

def load_data(train, selected_features=True):
    # import data
    if train:
        X_TRAIN_PATH = '../Case_material/train/X_train.csv'
        Y_TRAIN_PATH = '../Case_material/train/y_train.csv'
        X = pd.read_csv(X_TRAIN_PATH)
        y = pd.read_csv(Y_TRAIN_PATH).drop(columns=['ValueDateTimeUTC'])
    else:
        X_TEST_PATH = '../Case_material/test/X_test.csv'
        X = pd.read_csv(X_TEST_PATH)
        y = None


    # adding extra feature columns
    add_time(X)
    add_hour_dayofweek_month(X)
    add_hour_batches(X)
    add_weekend(X)
    add_business_hour(X)
    add_siesta(X)
    add_holidays_spain(X)
    add_city_weight(X)
    min_max_scale(X)

    if selected_features:  # only keeping feature columns selected by forward selection
        forward_selected_features = [
            "hour",
            "dayofweek",
            "dayofyear",
            "Madrid_t2m",
            "holidays",
            "Malaga_t2m"
        ]
        X = X[forward_selected_features]

    else:  # need to drop timestamp related features
        X.drop(columns=['ValueDateTimeUTC', 'time', 'date'], inplace=True)

    return (X, y)