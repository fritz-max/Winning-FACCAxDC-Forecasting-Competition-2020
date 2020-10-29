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
sns.set()


def report(results, n_top=3):
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


X_TRAIN_PATH = '../Case_material/train/X_train.csv'
Y_TRAIN_PATH = '../Case_material/train/y_train.csv'

X = pd.read_csv(X_TRAIN_PATH)
y = pd.read_csv(Y_TRAIN_PATH)

add_time(X)
add_hour_dayofweek_month(X)
add_hour_batches(X)
add_weekend(X)
add_business_hour(X)
add_siesta(X)
add_holidays_spain(X)
normalize(X)
add_city_weight(X)

cols2incl = [
    "hour",
    "dayofweek",
    "dayofyear",
    "Madrid_t2m",
    "holidays",
    "Malaga_t2m"
]


X = X[cols2incl]
# X.drop(columns=['ValueDateTimeUTC', 'time', 'date'], inplace=True)
y.drop(columns=['ValueDateTimeUTC'], inplace=True)


xgb_model = xgb.XGBRegressor(n_jobs=-1)
param_search = {
    'max_depth': randint(5, 20),
    'min_child_weight': [0.5, 1, 2],
    'learning_rate': loguniform(1e-4, 1e0),
    'gamma': [0,2,4,6,8,10]
    # 'subsample': [1],
    # 'colsample_bytree': [1]
}

split_frac = 0.9
X_train = X[:int(X.shape[0]*split_frac)]
X_test = X[int(X.shape[0]*split_frac):]
y_train = y[:int(X.shape[0]*split_frac)]
y_test = y[int(X.shape[0]*split_frac):]

tscv = TimeSeriesSplit(n_splits=20)
rsearch = RandomizedSearchCV(estimator=xgb_model, cv=tscv,
                             param_distributions=param_search)
rsearch.fit(X_train, y_train)


test_preds = rsearch.predict(X_test)

print('MSE:', round(mean_squared_error(y_test.to_numpy(),
                                       test_preds.reshape((-1, 1))), 2))
print('R^2:', round(r2_score(y_test.to_numpy(), test_preds.reshape((-1, 1))), 2))

mape = np.mean(np.abs((y_test.to_numpy() - test_preds.reshape((-1, 1))) /
                      np.abs(y_test.to_numpy())))
print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
print('Accuracy:', round(100*(1 - mape), 2))

report(rsearch.cv_results_)
