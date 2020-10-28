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
X_TRAIN_PATH = 'X_final.csv'
Y_TRAIN_PATH = 'y_train.csv'


X = pd.read_csv(X_TRAIN_PATH)
y = pd.read_csv(Y_TRAIN_PATH)

# X.drop(columns=['time', 'ValueDateTimeUTC'], inplace=True)
X.drop(columns=['ValueDateTimeUTC'], inplace=True)
y.drop(columns=['ValueDateTimeUTC'], inplace=True)
# print(X.head(), y.head())

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


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
