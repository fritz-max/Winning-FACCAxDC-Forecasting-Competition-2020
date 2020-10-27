import os
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv(os.getcwd() + '/predictions.csv', dtype='float')

    assert data.shape[0] == 8760, 'You have predicted either too few or too many samples, check dimensions of your predictions'
    assert data.shape[1] == 1, 'You have supplied too many columns'
    try:
        data.mean(skipna=True)
    except Exception as e:
        print('Incorrect data type for predictions due to %s, check your encoding' % e)

    print('Finished running evaluation of format, if no error arose then your predictions are most likely formatted correctly')

