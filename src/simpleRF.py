from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from pipelines import *
import numpy as np

## Loading in data
# TRAIN_PATH = '../Case_material/train/X_train.csv'

# Base model
# MSE = 7831015.162291523
# R2 = 0.6374358915697813
X_TRAIN_PATH = '../Case_material/train/X_train.csv'
Y_TRAIN_PATH = '../Case_material/train/y_train.csv'

# Better data
# MSE = 1653546.5784528889
# R2 = 0.9209651099261675
# X_TRAIN_PATH = 'X_final.csv'
# Y_TRAIN_PATH = 'y_train.csv'


X = pd.read_csv(X_TRAIN_PATH)
y = pd.read_csv(Y_TRAIN_PATH)

add_time(X)
add_hour_weekday_month(X)
add_weekend(X)
add_business_hour(X)
normalize(X)
add_city_weight(X, as_features=True)

cols2drop = ['time', 'ValueDateTimeUTC']
# cols2drop = ['time', 'ValueDateTimeUTC', 'Madrid_d2m', 'Madrid_t2m', 'Madrid_i10fg',
#              'Madrid_sp', 'Madrid_tcc', 'Madrid_tp', 'Barcelona_d2m',
#              'Barcelona_t2m', 'Barcelona_i10fg', 'Barcelona_sp', 'Barcelona_tcc',
#              'Barcelona_tp', 'Valencia_d2m', 'Valencia_t2m', 'Valencia_i10fg',
#              'Valencia_sp', 'Valencia_tcc', 'Valencia_tp', 'Seville_d2m',
#              'Seville_t2m', 'Seville_i10fg', 'Seville_sp', 'Seville_tcc',
#              'Seville_tp', 'Zaragoza_d2m', 'Zaragoza_t2m', 'Zaragoza_i10fg',
#              'Zaragoza_sp', 'Zaragoza_tcc', 'Zaragoza_tp', 'Malaga_d2m',
#              'Malaga_t2m', 'Malaga_i10fg', 'Malaga_sp', 'Malaga_tcc', 'Malaga_tp']

X.drop(columns=cols2drop, inplace=True)
# X.drop(columns=['ValueDateTimeUTC'], inplace=True)
y.drop(columns=['ValueDateTimeUTC'], inplace=True)
# print(X.head(), y.head())

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42)


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(
    n_estimators=100, random_state=42)
# Train the model on training data
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(mean_squared_error(y_test.to_numpy(), y_pred.reshape((-1, 1))))
print(r2_score(y_test.to_numpy(), y_pred.reshape((-1, 1))))

# add_time(df)
# add_hour_weekday_month(df)
# add_weekend(df)
# add_business_hour(df)
# normalize(df)
# add_city_weight(df)

# df.to_csv('./X_final.csv', sep=',')
