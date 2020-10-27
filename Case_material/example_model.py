import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def assign_datetimes(data):
    data = data.assign(ValueDateTimeUTC=lambda x: pd.to_datetime(x.ValueDateTimeUTC, format='%Y-%m-%d %H:%M', utc=True))
    data = data.set_index('ValueDateTimeUTC')
    return data


# Define root path to data (INSERT PATH HERE)
data_path = ""


# Import data sources
X_train = pd.read_csv(data_path + '/train/X_train.csv')
X_test = pd.read_csv(data_path + '/test/X_test.csv')
y_train = pd.read_csv(data_path + '/train/y_train.csv')


# Assign correct UTC timestamps to dataframes
X_train = assign_datetimes(X_train)
y_train = assign_datetimes(y_train)
X_test = assign_datetimes(X_test)


# Fit linear model
ols = LinearRegression()
ols.fit(X_train, y_train)


# Test to ensure predictions are formatted correctly
y_pred_train = ols.predict(X_train)
print('Mean absolute error is {}'.format(round(mean_absolute_error(y_train, y_pred_train), 2)))


# Create test predictions
y_pred_test = pd.DataFrame(ols.predict(X_test), columns=['Prediction'])
assert y_pred_test.shape[0] == X_test.shape[0]


# Save predictions to csv
y_pred_test.to_csv(data_path + '/predictions.csv', index=False)

