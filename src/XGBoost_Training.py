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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

X_TRAIN_PATH = '../Case_material/train/X_train.csv'
Y_TRAIN_PATH = '../Case_material/train/y_train.csv'

X_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH)

add_time(X_train)
add_hour_dayofweek_month(X_train)
add_hour_batches(X_train)
add_weekend(X_train)
add_business_hour(X_train)
add_siesta(X_train)
add_holidays_spain(X_train)
add_city_weight(X_train)
min_max_scale(X_train)

cols2incl = [
    "hour",
    "dayofweek",
    "dayofyear",
    "Madrid_t2m",
    "holidays",
    "Malaga_t2m"
    # "Malaga_d2m",
    # "weekofyear",
    # "hour3",
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

# X.drop(columns=['ValueDateTimeUTC', 'time', 'date'], inplace=True)
X_train = X_train[cols2incl]
y_train.drop(columns=['ValueDateTimeUTC'], inplace=True)

split_frac = 0.9
X_trainset = X_train[:int(X_train.shape[0]*split_frac)]
X_valset = X_train[int(X_train.shape[0]*split_frac):]
y_trainset = y_train[:int(X_train.shape[0]*split_frac)]
y_valset = y_train[int(X_train.shape[0]*split_frac):]

# pred5 FINAL


n_estimators = 389
params = {
    'learning_rate': 0.0562,
    'max_depth': 7,
    'min_child_weight': 1,
    'gamma': 0.5,
    'objective': 'reg:squarederror'
}


# pred 2

# 'learning_rate': 0.15687937227498,
# 'max_depth': 6,
# 'min_child_weight': 0.5,
# 'gamma': 1,
# Other parameters

xgb_model = XGBRegressor(
    **params,
    n_estimators=n_estimators,
    n_jobs=8)

# ============================================================
# UNCOMMENT FOR EVALUATION

# fit = xgb_model.fit(
#     X_trainset,
#     y_trainset,
#     eval_set=[(X_trainset, y_trainset), (X_valset, y_valset)],
#     early_stopping_rounds=25)

# test_preds = xgb_model.predict(X_valset)

# print('MSE:', round(mean_squared_error(y_valset.to_numpy(),
#                                        test_preds.reshape((-1, 1))), 2))
# print('R^2:', round(r2_score(y_valset.to_numpy(), test_preds.reshape((-1, 1))), 2))

# mape = np.mean(np.abs((y_valset.to_numpy() - test_preds.reshape((-1, 1))) /
#                       np.abs(y_valset.to_numpy())))
# print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
# print('Accuracy:', round(100*(1 - mape), 2))

# plt.figure(figsize=(16, 6))
# plt.plot(y_valset.to_numpy())
# plt.plot(test_preds)
# plt.show()

# ==============================================================
# UNCOMMENT FOR FINAL TRAINING AND MODEL SAVING

fit = xgb_model.fit(
    X_train,
    y_train)

fit.save_model('models/xgb_handin.model')
