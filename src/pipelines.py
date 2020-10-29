from datetime import date
import pandas as pd
from tqdm import tqdm


def _range(N, desc=''):
    return tqdm(range(N), desc=desc)


def add_time(df):
    df['time'] = pd.to_datetime(df['ValueDateTimeUTC'],
                                utc=True, infer_datetime_format=True)
    # df = df.drop(['ValueDateTimeUTC'], axis=1)
    # df.set_index('time', inplace=True)


def add_hour_dayofweek_month(df):
    # Generate 'hour', 'dayofweek' and 'month' features
    df.set_index('time', inplace=True)
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    df.reset_index(level=0, inplace=True)


def add_hour_batches(df):
    df['hour3'] = ((df.dayofweek == 0) |
                   (df.dayofweek == 1) | (df.dayofweek == 2))*1
    df['hour3'] = ((df.dayofweek == 3) |
                   (df.dayofweek == 4) | (df.dayofweek == 5))*2
    df['hour3'] = ((df.dayofweek == 6) |
                   (df.dayofweek == 7) | (df.dayofweek == 8))*3
    df['hour3'] = ((df.dayofweek == 9) | (df.dayofweek == 10) |
                   (df.dayofweek == 11))*4
    df['hour3'] = ((df.dayofweek == 12) |
                   (df.dayofweek == 13) | (df.dayofweek == 14))*5
    df['hour3'] = ((df.dayofweek == 15) |
                   (df.dayofweek == 16) | (df.dayofweek == 17))*6
    df['hour3'] = ((df.dayofweek == 18) |
                   (df.dayofweek == 19) | (df.dayofweek == 20))*7
    df['hour3'] = ((df.dayofweek == 21) |
                   (df.dayofweek == 22) | (df.dayofweek == 23))*8


def add_siesta(df):
    df['siesta'] = ((df.hour == 14) | (df.hour == 15) | (df.hour == 16))


def add_holidays_spain(df):
    holidays_leap_year = [1, 6, 122, 228, 286, 306, 341, 343, 360]
    holidays = [1, 6, 121, 227, 285, 305, 340, 342, 359]
    df['year'] = df['date'].dt.year
    df['holiday'] = df.dayofyear.isin(
        holidays) & df.year.isin([2015, 2017, 2018, 2019])
    df['holiday'] += df.dayofyear.isin(holidays_leap_year) & (df.year == 2016)
    df.drop(columns=['year'], inplace=True)


def before_holidays_spain(df):
    df.reset_index(level=0, inplace=True)

    for a in range(int((len(df)-48)/24)):
        a1 = a+1
        a2 = a+2
        if a == 0:
            for i in range(24):
                df.loc[i, 'before_holidays'] = 0
                continue

        else:
            for i, i1 in zip(range(a*24, a1*24), range(a1*24, a2*24)):
                if df.loc[i1, "holidays"] == 1 and df.loc[i, "holidays"] != 1:
                    df.loc[i, 'before_holidays'] = 1
                else:
                    df.loc[i, 'before_holidays'] = 0
                continue

    for i in range(len(df)-49, len(df)):
        df.loc[i, 'before_holidays'] = 0
        continue


def add_weekend(df):
    # assert df.dayofweek, "run add_hour_dayofweek_month before running this"
    # Generate 'weekend' feature
    df['weekend'] = (df.dayofweek == 5)*2
    df['weekend'] += (df.dayofweek == 6)*1


def add_business_hour(df):
    # Generate 'business hour' feature
    df.set_index('time', inplace=True)
    hour = df.hour
    df['business hour'] = (((hour > 8) & (hour < 14))
                           | ((hour > 16) & (hour < 21)))*2
    df['business hour'] |= ((hour >= 14) & (hour <= 16))*1
    df.reset_index(level=0, inplace=True)


def add_avgs(df):
    '''script for creating a dataset only comprised of the average of the different types of features for the cities'''
    d2ms = ['Madrid_d2m', 'Barcelona_d2m', 'Valencia_d2m',
            'Seville_d2m', 'Zaragoza_d2m', 'Malaga_d2m']

    t2ms = ['Madrid_t2m', 'Barcelona_t2m', 'Valencia_t2m',
            'Seville_t2m', 'Zaragoza_t2m', 'Malaga_t2m']

    i10fgs = ['Madrid_i10fg', 'Barcelona_i10fg', 'Valencia_i10fg',
              'Seville_i10fg', 'Zaragoza_i10fg', 'Malaga_i10fg']

    sps = ['Madrid_sp', 'Barcelona_sp', 'Valencia_sp',
           'Seville_sp', 'Zaragoza_sp', 'Malaga_sp']

    tccs = ['Madrid_tcc', 'Barcelona_tcc', 'Valencia_tcc',
            'Seville_tcc', 'Zaragoza_tcc', 'Malaga_tcc']

    tps = ['Madrid_tp', 'Barcelona_tp', 'Valencia_tp',
           'Seville_tp', 'Zaragoza_tp', 'Malaga_tp']

    df['d2m'] = df[d2ms].mean(axis=1).round(2)
    df['t2m'] = df[t2ms].mean(axis=1).round(2)
    df['i10fg'] = df[i10fgs].mean(axis=1).round(2)
    df['sp'] = df[sps].mean(axis=1).round(2)
    df['tcc'] = df[tccs].mean(axis=1).round(2)
    df['tps'] = df[tps].mean(axis=1).round(2)


def add_city_weight(df, as_features=False):
    # Calculate the weight of every city
    total_pop = 6155116 + 5179243 + 2541000 + 1950000 + 783763 + 600000

    weight_Madrid = 6155116 / total_pop
    weight_Barcelona = 5179243 / total_pop
    weight_Valencia = 2541000 / total_pop
    weight_Seville = 1950000 / total_pop
    weight_Zaragoza = 783763 / total_pop  # not very accurate
    # not very accurate, only had one from 2018, which was 571026
    weight_Malaga = 600000 / total_pop

    if as_features:
        df['Madrid_w'] = weight_Madrid
        df['Barcelona_w'] = weight_Barcelona
        df['Valencia_w'] = weight_Valencia
        df['Zaragoza_w'] = weight_Zaragoza
        df['Malaga_w'] = weight_Malaga
    else:
        Madrid = ['Madrid_d2m', 'Madrid_t2m', 'Madrid_i10fg',
                  'Madrid_sp', 'Madrid_tcc', 'Madrid_tp']
        Barcelona = ['Barcelona_d2m', 'Barcelona_t2m', 'Barcelona_i10fg',
                     'Barcelona_sp', 'Barcelona_tcc', 'Barcelona_tp']
        Valencia = ['Valencia_d2m', 'Valencia_t2m', 'Valencia_i10fg',
                    'Valencia_sp', 'Valencia_tcc', 'Valencia_tp']
        Seville = ['Seville_d2m', 'Seville_t2m', 'Seville_i10fg',
                   'Seville_sp', 'Seville_tcc', 'Seville_tp']
        Zaragoza = ['Zaragoza_d2m', 'Zaragoza_t2m', 'Zaragoza_i10fg',
                    'Zaragoza_sp', 'Zaragoza_tcc', 'Zaragoza_tp']
        Malaga = ['Malaga_d2m', 'Malaga_t2m', 'Malaga_i10fg',
                  'Malaga_sp', 'Malaga_tcc', 'Malaga_tp']

        cities = [Madrid, Barcelona, Valencia, Seville, Zaragoza, Malaga]
        weights = [weight_Madrid, weight_Barcelona, weight_Valencia,
                   weight_Seville, weight_Zaragoza, weight_Malaga]

        for (w, c) in zip(weights, cities):
            df[c] *= w


def _norm(value, max, min):
    return (value - min)/(max-min)


def normalize(df):
    d2ms = ['Madrid_d2m', 'Barcelona_d2m', 'Valencia_d2m',
            'Seville_d2m', 'Zaragoza_d2m', 'Malaga_d2m']

    t2ms = ['Madrid_t2m', 'Barcelona_t2m', 'Valencia_t2m',
            'Seville_t2m', 'Zaragoza_t2m', 'Malaga_t2m']

    i10fgs = ['Madrid_i10fg', 'Barcelona_i10fg', 'Valencia_i10fg',
              'Seville_i10fg', 'Zaragoza_i10fg', 'Malaga_i10fg']

    sps = ['Madrid_sp', 'Barcelona_sp', 'Valencia_sp',
           'Seville_sp', 'Zaragoza_sp', 'Malaga_sp']

    tccs = ['Madrid_tcc', 'Barcelona_tcc', 'Valencia_tcc',
            'Seville_tcc', 'Zaragoza_tcc', 'Malaga_tcc']

    tps = ['Madrid_tp', 'Barcelona_tp', 'Valencia_tp',
           'Seville_tp', 'Zaragoza_tp', 'Malaga_tp']

    types = [d2ms, t2ms, i10fgs, sps, tccs, tps]

    for type in types:
        max, min = df[type].max().max(), df[type].min().min()
        def curried_norm(val): return _norm(val, max, min)
        df[type] = df[type].apply(curried_norm)


def bin_cloud_coverage(df):
    def bin(x):
        if x > 0.8:
            return 1
        else:
            return 0
    for col in df.columns:
        if col.endswith("tcc"):
            df[col+"_binary"] = df[col].apply(bin)
        else:
            continue


def series_to_supervised(df, n_in=1, n_out=1, dropnan=True):
    n_vars = df.shape[1]
    # df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
# ### Example of lagging by 24h ####
# ### input shape (26304, 43) ###
# train_series = series_to_supervised(train_data,24,1,True)
# ### output shape (26280, 1076) ###
