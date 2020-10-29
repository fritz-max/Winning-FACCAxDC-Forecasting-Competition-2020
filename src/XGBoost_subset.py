from pipelines import *
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBRFRegressor, DMatrix, cv
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import uniform, loguniform, randint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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
add_city_weight(X)
normalize(X)

cols2incl = [
    "hour",
    "dayofweek",
    "dayofyear",
    "Madrid_t2m",
    "holidays",
    "Malaga_t2m",
    "Malaga_d2m",
    "weekofyear",
    "hour3"
    # "Seville_t2m",
    # "Seville_tp",
    # "quarter",
    # "Zaragoza_t2m",
    # "Valencia_tp",
    # "Barcelona_tcc",
    # "dayofmonth",
    # "Madrid_tcc",
    # "Barcelona_i10fg",
    # "weekend",
    # "month",
    # "Zaragoza_d2m",
    # "Madrid_i10fg",
    # "Seville_tcc",
    # "Barcelona_tp",
    # "Malaga_sp",
]

cols2drop = ['time', 'ValueDateTimeUTC', 'Madrid_d2m', 'Madrid_t2m', 'Madrid_i10fg',
             'Madrid_sp', 'Madrid_tcc', 'Madrid_tp', 'Barcelona_d2m',
             'Barcelona_t2m', 'Barcelona_i10fg', 'Barcelona_sp', 'Barcelona_tcc',
             'Barcelona_tp', 'Valencia_d2m', 'Valencia_t2m', 'Valencia_i10fg',
             'Valencia_sp', 'Valencia_tcc', 'Valencia_tp', 'Seville_d2m',
             'Seville_t2m', 'Seville_i10fg', 'Seville_sp', 'Seville_tcc',
             'Seville_tp', 'Zaragoza_d2m', 'Zaragoza_t2m', 'Zaragoza_i10fg',
             'Zaragoza_sp', 'Zaragoza_tcc', 'Zaragoza_tp', 'Malaga_d2m',
             'Malaga_t2m', 'Malaga_i10fg', 'Malaga_sp', 'Malaga_tcc', 'Malaga_tp']


# X.drop(columns=['ValueDateTimeUTC', 'time', 'date'], inplace=True)
X = X[cols2incl]
y.drop(columns=['ValueDateTimeUTC'], inplace=True)

# xgb_model = xgb.XGBRegressor()
# param_search = {
#     'max_depth': randint(5, 20),
#     'learning_rate': loguniform(1e-4, 1e0)}

split_frac = 0.9
X_train = X[:int(X.shape[0]*split_frac)]
X_test = X[int(X.shape[0]*split_frac):]
y_train = y[:int(X.shape[0]*split_frac)]
y_test = y[int(X.shape[0]*split_frac):]

n_estimators = 2000
# params = {
#     # Parameters that we are going to tune.
#     'learning_rate': 0.13872832647046318,
#     'max_depth': 7,
#     # Other parameters
#     'objective': 'reg:squarederror'
# }

# xgb_model = XGBRegressor(
#     **params,
#     n_estimators=n_estimators,
#     n_jobs=8)

split_frac = 0.9
X_train = X[:int(X.shape[0]*split_frac)]
X_test = X[int(X.shape[0]*split_frac):]
y_train = y[:int(X.shape[0]*split_frac)]
y_test = y[int(X.shape[0]*split_frac):]

i = 1
score = []

rf = RandomForestRegressor(
    n_estimators=1000, random_state=42)
# Train the model on training data
fit = rf.fit(X_train, y_train)

test_preds = fit.predict(X_test)

# fit = xgb_model.fit(
#     X_train,
#     y_train,
#     eval_set=[(X_train, y_train), (X_test, y_test)],
#     early_stopping_rounds=25)


# test_preds = xgb_model.predict(X_test)

print('MSE:', round(mean_squared_error(y_test.to_numpy(),
                                       test_preds.reshape((-1, 1))), 2))
print('R^2:', round(r2_score(y_test.to_numpy(), test_preds.reshape((-1, 1))), 2))

mape = np.mean(np.abs((y_test.to_numpy() - test_preds.reshape((-1, 1))) /
                      np.abs(y_test.to_numpy())))
print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
print('Accuracy:', round(100*(1 - mape), 2))
