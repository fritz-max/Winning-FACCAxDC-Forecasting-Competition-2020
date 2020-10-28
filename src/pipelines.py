import pandas as pd

def add_hour_weekday_month(df):
    # Generate 'hour', 'weekday' and 'month' features
    for i in range(len(df)):
        position = df.index[i]
        hour = position.hour
        weekday = position.weekday()
        month = position.month
        df.loc[position, 'hour'] = hour
        df.loc[position, 'weekday'] = weekday
        df.loc[position, 'month'] = month


def add_weekend(df):
    assert df.weekday, "run add_hour_weekday_month before running this"
    # Generate 'weekend' feature
    for i in range(len(df)):
        position = df.index[i]
        weekday = position.weekday()
        if (weekday == 6):
            df.loc[position, 'weekday'] = 2
        elif (weekday == 5):
            df.loc[position, 'weekday'] = 1
        else:
            df.loc[position, 'weekday'] = 0


def add_business_hour(df):
    # Generate 'business hour' feature
    for i in range(len(df)):
        position = df.index[i]
        hour = position.hour
        if ((hour > 8 and hour < 14) or (hour > 16 and hour < 21)):
            df.loc[position, 'business hour'] = 2
        elif (hour >= 14 and hour <= 16):
            df.loc[position, 'business hour'] = 1
        else:
            df.loc[position, 'business hour'] = 0

def add_time(df):
    df = add_hour_weekday_month(df)
    df = add_weekend(df)
    df = add_business_hour(df)
    return df

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


def add_city_weight(df):
    # Calculate the weight of every city
    total_pop = 6155116 + 5179243 + 1645342 + 783763 + 600000

    weight_Madrid = 6155116 / total_pop
    weight_Barcelona = 5179243 / total_pop
    weight_Valencia = 1645342 / total_pop
    weight_Zaragoza = 783763 / total_pop  # not very accurate
    # not very accurate, only had one from 2018, which was 571026
    weight_Malaga = 600000 / total_pop

    df['Madrid_w'] = weight_Madrid
    df['Barcelona_w'] = weight_Barcelona
    df['Valencia_w'] = weight_Valencia
    df['Zaragoza_w'] = weight_Zaragoza
    df['Malaga_w'] = weight_Malaga

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
        curried_norm = lambda val : _norm(val, max, min)
        df[type] = df[type].apply(curried_norm)

