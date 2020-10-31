from numpy.core.numeric import load
from preprocessing import load_data
from feature_engineering import *
from xgboost import XGBRegressor
import pandas as pd

prediction_path = "../Case_material/predictions/predictions.csv"

(X_test, _) = load_data(train=False)

xgb_model = XGBRegressor(n_jobs=8)
xgb_model.load_model("models/xgb_handin.model")

predictions = xgb_model.predict(X_test)
pd.DataFrame({"Prediction": predictions}).to_csv(prediction_path, sep=",", index=False)

print("First five predictions:", predictions[:5]) 
print("The predictions were place in:", prediction_path)
