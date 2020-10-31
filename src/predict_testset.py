from pipelines import *
from xgboost import XGBRegressor, XGBRFRegressor, DMatrix, cv
import pandas as pd

X_TEST_PATH = '../Case_material/test/X_test.csv'
X_test = pd.read_csv(X_TEST_PATH)

add_time(X_test)
add_hour_dayofweek_month(X_test)
add_hour_batches(X_test)
add_weekend(X_test)
add_business_hour(X_test)
add_siesta(X_test)
add_holidays_spain(X_test)
add_city_weight(X_test)
min_max_scale(X_test)

features = [
    "hour",
    "dayofweek",
    "dayofyear",
    "Madrid_t2m",
    "holidays",
    "Malaga_t2m"
]

X_test = X_test[cols2incl]

xgb_model = XGBRegressor(n_jobs=8)
xgb_model.load_model("models/xgb_handin.model")

test_preds = xgb_model.predict(X_test)

print(test_preds[:5]) 

filename = "predictions5"

pd.DataFrame({"Prediction": test_preds}).to_csv(f"../Case_material/predictions/{filename}.csv", sep=",", index=False)
