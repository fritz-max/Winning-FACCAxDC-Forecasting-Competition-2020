import numpy as np
import pandas as pd
from xgboost import XGBRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.decomposition import PCA

class XGB_per_hour:
    def __init__(self):
        self.models = list()
        for i in range(24):
            xgb_model = XGBRegressor()
            self.models.append(xgb_model)
        
    def fit(self, X_train, y_train):
        for i in range(24):
            print("Fitting model ", i)
            hX_train = X_train.loc[X_train['hour'] == i]
            hy_train = y_train.loc[X_train['hour'] == i]
            self.models[i].fit(hX_train, hy_train)
    
    def evaluate(self, X_test, y_test):        
        mses = []
        mapes = []
        mape_accs = []

        for i in range(24):
            print("Evaluating Model ", i)
            hX_test = X_test.loc[X_test['hour'] == i]
            hy_test = y_test.loc[X_test['hour'] == i]
            
            h_preds = self.models[i].predict(hX_test)

            mse = round(mean_squared_error(hy_test.to_numpy(), h_preds.reshape((-1, 1))),2)
            mape = np.mean(np.abs((hy_test.to_numpy() - h_preds.reshape((-1,1))) / np.abs(hy_test.to_numpy())))
            mape_acc = round(100*(1 - mape), 2)

            mses.append(mse)
            mapes.append(mape)
            mape_accs.append(mape_acc)
            
        return np.mean(mses), np.mean(mapes), np.mean(mape_accs)

if __name__ == "__main__":
        
    X_TRAIN_PATH = './X_final.csv'
    Y_TRAIN_PATH = './y_train.csv'

    X = pd.read_csv(X_TRAIN_PATH)
    y = pd.read_csv(Y_TRAIN_PATH)

    X.drop(columns=['ValueDateTimeUTC', 'time'], inplace=True)
    # X.drop(columns=['ValueDateTimeUTC'], inplace=True)
    y.drop(columns=['ValueDateTimeUTC'], inplace=True)

    # pca = PCA(n_components=25)
    # pca.fit(X)
    # X = pca.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    xgb_ph = XGB_per_hour()

    xgb_ph.fit(X_train, y_train)

    mse, mape, acc = xgb_ph.evaluate(X_test, y_test)

    print("\n===================")
    print("MSE: ", mse)
    print("MAPE: ", mape)
    print("Acc.: ", acc)