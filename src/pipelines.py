from datetime import date
import pandas as pd

def add_time(df):
    df['time'] = pd.to_datetime(df['ValueDateTimeUTC'],
                                utc=True, infer_datetime_format=True)


def add_hour_dayofweek_month(df):
    '''Generate time related features
    '''
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



def add_siesta(df):
    df['siesta'] = ((df.hour == 14) | (df.hour == 15) | (df.hour == 16))

def add_hour_batches(df):
    assert 'hour' in df, "run add_hour_dayofweek_month before running this"
    df['hour3'] = ((df.hour== 0) | (df.hour== 1) | (df.hour== 2))*1
    df['hour3'] = ((df.hour== 3) | (df.hour== 4) | (df.hour== 5))*2
    df['hour3'] = ((df.hour== 6) | (df.hour== 7) | (df.hour== 8))*3
    df['hour3'] = ((df.hour== 9) | (df.hour== 10) | (df.hour== 11))*4
    df['hour3'] = ((df.hour== 12) | (df.hour== 13) | (df.hour== 14))*5
    df['hour3'] = ((df.hour== 15) | (df.hour== 16) | (df.hour== 17))*6
    df['hour3'] = ((df.hour== 18) | (df.hour== 19) | (df.hour== 20))*7
    df['hour3'] = ((df.hour== 21) | (df.hour== 22) | (df.hour== 23))*8

def add_holidays_spain(df):
    holidays_leap_year = [1, 6, 122, 228, 286, 306, 341, 343, 360]
    holidays = [1, 6, 121, 227, 285, 305, 340, 342, 359]
    df['year'] = df['date'].dt.year
    df['holidays'] = df.dayofyear.isin(
        holidays) & df.year.isin([2015, 2017, 2018, 2019])
    df['holidays'] += df.dayofyear.isin(holidays_leap_year) & (df.year == 2016)
    df.drop(columns=['year'], inplace=True)



def add_weekend(df):
    assert 'dayofweek' in df, "run add_hour_dayofweek_month before running this"
    # Generate 'weekend' feature
    df['weekend'] = (df.dayofweek == 5)*2
    df['weekend'] += (df.dayofweek == 6)*1


def add_business_hour(df):
    assert 'hour' in df, "run add_hour_dayofweek_month before running this"
    # Generate 'business hour' feature
    df.set_index('time', inplace=True)
    hour = df.hour
    df['business hour'] = (((hour > 8) & (hour < 14))
                           | ((hour > 16) & (hour < 21)))*2
    df['business hour'] |= ((hour >= 14) & (hour <= 16))*1
    df.reset_index(level=0, inplace=True)


def add_avgs(df):
    '''script for creating a dataset only comprised of the average of the
    different types of features for the cities
    '''
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


def add_city_weight(df, as_columns=False):
    '''Calculate the weight of every city
    '''
    total_pop = 6155116 + 5179243 + 2541000 + 1950000 + 783763 + 600000

    weight_Madrid = 6155116 / total_pop
    weight_Barcelona = 5179243 / total_pop
    weight_Valencia = 2541000 / total_pop
    weight_Seville = 1950000 / total_pop
    weight_Zaragoza = 783763 / total_pop  # not very accurate
    # not very accurate, only had one from 2018, which was 571026
    weight_Malaga = 600000 / total_pop

    if as_columns:
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


def _min_max_scale(value, max, min):
    return (value - min)/(max-min)


def min_max_scale(df):
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
        curried_min_max = lambda val: _min_max_scale(val, max, min)
        df[type] = df[type].apply(curried_min_max)


def bin_cloud_coverage(df, threshold):
    tccs = ['Madrid_tcc', 'Barcelona_tcc', 'Valencia_tcc',
        'Seville_tcc', 'Zaragoza_tcc', 'Malaga_tcc']
    df[tccs] = (df[tccs] > threshold).astype('bool')

