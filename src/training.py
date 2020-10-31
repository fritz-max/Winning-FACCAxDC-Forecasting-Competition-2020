from numpy.core.numeric import load
from preprocessing import load_data
from feature_engineering import *
from xgboost import XGBRegressor
import pandas as pd


(X_train, y_train) = load_data(train=True)


# setting up regressor model with hyper parameters
params = {
    'learning_rate': 0.0562,
    'max_depth': 7,
    'min_child_weight': 1,
    'gamma': 0.5,
    'objective': 'reg:squarederror',
    'n_estimators': 389,
}

xgb_model = XGBRegressor(
    **params,
    n_jobs=-1)

# training xgb regressor using xgboost's sklearn api
print('Training model...')
reg = xgb_model.fit(X_train,y_train)
reg.save_model('models/xgb_handin.model')
print('Model was saved: ./models/xgb_handin.model')
