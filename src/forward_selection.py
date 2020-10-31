from sys import path
path.append("../")
from pipelines import *
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBRFRegressor, DMatrix, cv
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import numpy as np
from scipy.stats import uniform, loguniform, randint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import itertools
sns.set()

X_TRAIN_PATH = '../../Case_material/train/X_train.csv'
Y_TRAIN_PATH = '../../Case_material/train/y_train.csv'

X = pd.read_csv(X_TRAIN_PATH)
y = pd.read_csv(Y_TRAIN_PATH)

add_time(X)
add_hour_dayofweek_month(X)
add_hour_batches(X)
add_weekend(X)
add_business_hour(X)
add_siesta(X)
add_holidays_spain(X)
add_city_weight(X)
min_max_scale(X)

X.drop(columns=['ValueDateTimeUTC', 'time', 'date'], inplace=True)
y.drop(columns=['ValueDateTimeUTC'], inplace=True)

split_frac = 0.9
X_train = X[:int(X.shape[0]*split_frac)]
X_test = X[int(X.shape[0]*split_frac):]
y_train = y[:int(X.shape[0]*split_frac)]
y_test = y[int(X.shape[0]*split_frac):]

n_estimators = 2000
params = {
    # Parameters that we are going to tune.
    'learning_rate': 0.13872832647046318,
    'max_depth': 7,
    # Other parameters
    'objective': 'reg:squarederror'
}

split_frac = 0.9
X_train = X[:int(X.shape[0]*split_frac)]
X_test = X[int(X.shape[0]*split_frac):]
y_train = y[:int(X.shape[0]*split_frac)]
y_test = y[int(X.shape[0]*split_frac):]

i = 1
score = []

#  = pd.DataFrame(columns = ['model', 'features', 'MSE', 'R_squared'])

model_computations = 0
starting_time = time.time()

included_features = [] # features already included in the selection (at start: no features yet)
remaining_features = list(X_train.columns.values) # features to be included

results = []
metrics = []

# loop over the amount of all features
for k in range(1, len(X_train.columns)+1):

    best_MSE = np.inf

    # update remaining features
    remaining_features = [p for p in X_train.columns if p not in included_features]
    
    for feature in itertools.combinations(remaining_features, 1):

        print("=====================================================================================")
        print("Loop: ", k)
        print("Trying Feature: ", feature)
        print("Complete List: ", list(feature)+included_features)
        print("=====================================================================================")

        X_train_subset = X_train[list(feature)+included_features]
        X_test_subset = X_test[list(feature)+included_features]
        
        xgb_model = XGBRegressor(
            **params,
            n_estimators=n_estimators,
            n_jobs=6)

        fit = xgb_model.fit(
            X_train_subset,
            y_train,
            eval_set=[(X_train_subset, y_train), (X_test_subset, y_test)],
            early_stopping_rounds=25)
                
        # Calculate RSS for this specific model an append it to results
        MSE = mean_squared_error(y_test, fit.predict(X_test_subset))
        
        if MSE < best_MSE:
            best_model = fit
            best_MSE = MSE
            # best_R_sq = R_squared
            best_feature = feature[0]
                
        model_computations += 1
    
    # Update included features
    included_features.append(best_feature)
    remaining_features.remove(best_feature)
    
    print('Completed computation for', k, 'features.')
    print("included features: ", included_features)
    
    results.append({'num_f': k, 'model':best_model, 'features':best_feature, 'MSE':best_MSE, 'included_feat': included_features})
    metrics.append({'num_f': k, 'Best MSE':best_MSE, 'included_feat': included_features})

    f = open(f"metrics{k}.txt", "w")
    f.write(f"num: {k}, \nBest MSE: {best_MSE}, \nincluded:")
    for e in included_features:
        f.write("\n")
        f.writelines(e)
    f.close()

end_time = time.time()

print('Done!')
print('Number of computed models: ', model_computations)
print('Total computation time: ', round(end_time-starting_time, 2))

print("metrics: ", metrics)
print("Results: ", results)

