## Imports
import pandas as pd
from pipelines import normalize

## Loading in data
TRAIN_PATH = '../Case_material/train/X_train.csv'
df = pd.read_csv(TRAIN_PATH)

normalize(df)
print(df.describe())