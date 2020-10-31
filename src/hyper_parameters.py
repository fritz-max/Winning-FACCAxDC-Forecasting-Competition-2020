from feature_engineering import *
from preprocessing import load_data
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import numpy as np
from scipy.stats import uniform, loguniform, randint


def report(results, n_top=3):
    '''https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
    '''
    # Utility function to report best scores
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# load preprocessed data
(X, y) = load_data(train=True)

xgb_model = xgb.XGBRegressor()
param_search = {
    'max_depth': randint(5, 20),
    'min_child_weight': uniform(1.5, 1),
    'learning_rate': loguniform(1e-4, 1e0),
    'gamma': uniform(5, 5)
}

tscv = TimeSeriesSplit(n_splits=10)

rsearch = RandomizedSearchCV(
    estimator=xgb_model,
    cv=tscv,
    param_distributions=param_search,
    verbose=True,
    n_jobs=-1
).fit(X, y)

report(rsearch.cv_results_)
