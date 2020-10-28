'''script for creating a dataset only comprised of the average of the different types of features for the cities'''

## Imports
import pandas as pd
from pipelines import add_avgs

## Loading in data
TRAIN_PATH = '../Case_material/train/X_train.csv'
df = pd.read_csv(TRAIN_PATH)

print(len(df.columns))
## Preprocessing pipeline
add_avgs(df)

print(len(df.columns))

columns_to_drop = ['Madrid_d2m', 'Madrid_t2m', 'Madrid_i10fg',
       'Madrid_sp', 'Madrid_tcc', 'Madrid_tp', 'Barcelona_d2m',
       'Barcelona_t2m', 'Barcelona_i10fg', 'Barcelona_sp', 'Barcelona_tcc',
       'Barcelona_tp', 'Valencia_d2m', 'Valencia_t2m', 'Valencia_i10fg',
       'Valencia_sp', 'Valencia_tcc', 'Valencia_tp', 'Seville_d2m',
       'Seville_t2m', 'Seville_i10fg', 'Seville_sp', 'Seville_tcc',
       'Seville_tp', 'Zaragoza_d2m', 'Zaragoza_t2m', 'Zaragoza_i10fg',
       'Zaragoza_sp', 'Zaragoza_tcc', 'Zaragoza_tp', 'Malaga_d2m',
       'Malaga_t2m', 'Malaga_i10fg', 'Malaga_sp', 'Malaga_tcc', 'Malaga_tp']

df.drop(columns=columns_to_drop, inplace=True)


print(df.columns)
print(len(df.columns))