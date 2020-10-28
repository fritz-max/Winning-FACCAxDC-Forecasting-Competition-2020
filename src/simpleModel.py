## Imports
import pandas as pd
from pipelines import normalize
import numpy as np

def split_data(series, train_fraq, test_len=8760):
    """Splits input series into train, val and test.
    
        Default to 1 year of test data.
    """
    #slice the last year of data for testing 1 year has 8760 hours
    test_slice = len(series)-test_len

    test_data = series[test_slice:]
    train_val_data = series[:test_slice]

    #make train and validation from the remaining
    train_size = int(len(train_val_data) * train_fraq)
    
    train_data = train_val_data[:train_size]
    val_data = train_val_data[train_size:]
    
    return train_data, val_data, test_data
## Loading in data
TRAIN_PATH = '../Case_material/train/X_train.csv'
df = pd.read_csv(TRAIN_PATH)

normalize(df)
train_multi, val_multi, test_multi = split_data(df, train_fraq=0.65, test_len=0)
print()
print("Multivarate Datasets")
print(f"Train Data Shape: {train_multi.shape}")
print(f"Val Data Shape: {val_multi.shape}")
print(f"Test Data Shape: {test_multi.shape}")
print(f"Nulls In Train {np.any(pd.isnull(train_multi))}")
print(f"Nulls In Validation {np.any(pd.isnull(val_multi))}")
print(f"Nulls In Test {np.any(pd.isnull(test_multi))}")
print()

