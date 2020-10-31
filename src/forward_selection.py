from preprocessing import load_data
from feature_engineering import *
from sklearn.metrics import mean_squared_error
import pandas as pd
import xgboost as xgb
import numpy as np
import numpy as np
import time
import itertools

(X, y) = load_data(train=True, selected_features=False)

split_frac = 0.9
X_train = X[:int(X.shape[0]*split_frac)]
X_test = X[int(X.shape[0]*split_frac):]
y_train = y[:int(X.shape[0]*split_frac)]
y_test = y[int(X.shape[0]*split_frac):]

params = {
    'learning_rate': 0.0562,
    'max_depth': 7,
    'min_child_weight': 1,
    'gamma': 0.5,
    'objective': 'reg:squarederror',
    'n_estimators': 1000,  # should have found solution before 1000
}

model_computations = 0
starting_time = time.time()

included_features = []  # features already included in the selection (at start: no features yet)
remaining_features = list(X_train.columns.values)  # features to be included

results = []
metrics = []
best_feature = None
best_model = None

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
        
        xgb_model = xgb.XGBRegressor(
            **params,
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

