#### meaning of weather params ###
# i10fg - Instantaneous 10 metre wind gust
# sp - Surface pressure
# tcc - Total cloud cover
# tp - Total precipitation
# d2m - 2 metre dewpoint temperature 
# t2m - 2 metre temperature 
### Plotting the corr matrix ###
####### Clouds binary func cutting at 0.8 #######

def bin(x):
    if x > 0.8:
        return 1
    else:
        return 0 
def binary_tcc(data):
    df = data
    for col in data.columns:
        if col.endswith("tcc"):
            df[col] = df[col].apply(bin)
    return df
############################################################
def weighted_avg_ds(data, weight_col=False, weight_inplace=False,
                    weights_dict={"Madrid":1/6,"Barcelona":1/6,"Valencia":1/6,"Seville":1/6,"Zaragoza":1/6,"Malaga":1/6 }):
    
    """This func will transfrom the full dataset to any dimension we would like
    Weights for each city  need to be specified in the weights_dict, if weights are equal (1/6), it will just average everything out normally
    If weight_col =  True, it will add 6 separate columns with choosen weights
    If weight_inplace = True it will multiply the weights by all the features for each city separately, but not for the target var 
    Binary cloud cover transformation is factored in this function regardless of the args passed
    """
    col_ends = ["_d2m", "_t2m","_i10fg", "_sp", "_tcc", "_tp"]    
    df = pd.DataFrame()
    df["time"] = data["ValueDateTimeUTC"]
    #print(df)
    #### Adding the power consumption col if present in data ### 
    if 'VolumeMWh' in data.columns:
        df['VolumeMWh'] = data['VolumeMWh']
        
    if weight_inplace == False:
    
        if weight_col == False:

            for col in col_ends:
                a = 0 
                
                if not col.endswith("tcc"):
                    for city in weights_dict.keys():

                        a+=1
                        if a == 1:

                            df[col[1:]] = data[city+col]*(weights_dict[city])
                            continue
                        colum = data[city+col]*weights_dict[city]
                        df[col[1:]] = df[col[1:]]+colum
                        
                elif col.endswith("tcc"):
                    
                    for city in weights_dict.keys():

                        a+=1
                        if a == 1:

                            df[col[1:]] = data[city+col].apply(bin) * weights_dict[city]
                            continue
                        colum = data[city+col].apply(bin) * weights_dict[city]
                        df[col[1:]] = df[col[1:]]+colum
                    ### rounding tcc to 0 or 1 ####     
                    df[col[1:]] = df[col[1:]].apply(round).astype("int64")              
        elif weight_col == True:
            df = data
            df["time"] = data["ValueDateTimeUTC"] 
            df = df.drop(["ValueDateTimeUTC"],inplace=False,axis=1)
            for city in weights_dict.keys():
                df[city+"_weight"] = round(weights_dict[city],3)
            ## apply the binary col for all the cols ending with tcc ###
            for col in df.columns:
                if col.endswith("tcc"):
                    df[col]=df[col].apply(bin)
                    
    elif weight_inplace == True and weight_col==False:
        
         for city in weights_dict.keys():
            for column in data.columns:
                if column.startswith(city):
                    if column.endswith("tcc"):
                        df[column] = data[column].apply(bin)
                        continue
                    df[column] = data[column]*weights_dict[city]
                    continue
    
    return df.set_index("time")

##### Siesta func ####
def siesta(h,hours=[14,15,16]):
    if h in hours:
        return 1
    else:
        return 0 

### make time features func ####
def make_time_features(data, series):
    df = data.reset_index(drop=True)
    
    ### remove timestamp cols if they are still in data at this point anyhow ###
    if "time" in df.columns:
        df = df.drop("time")
    if "ValueDateTimeUTC" in df.columns:
        df = df.drop("ValueDateTimeUTC")
    #convert series to datetimes
    times = series.apply(lambda x: x.split('+')[0])
    datetimes = pd.DatetimeIndex(times)
    
    hours = datetimes.hour.values
    day = datetimes.dayofweek.values
    months = datetimes.month.values
    
    hour = pd.Series(hours, name='hours')
    dayofw = pd.Series(day, name='dayofw')
    month = pd.Series(months, name='months')
    ### printing to check for NANs, if there are any
    ### then theres an error
    print(hour[:5])
    df["hour"] = hour
    df["dayofw"] = dayofw
    df["months"] = month
    return df



 #- average not weighted 6 x N Done
train_X_avg = weighted_avg_ds(X_train, weight_col=False, weight_inplace=False, weights_dict={"Madrid":1/6,"Barcelona":1/6,"Valencia":1/6,"Seville":1/6,"Zaragoza":1/6,"Malaga":1/6 })
#transforming index to time features
train_X_avg = make_time_features(train_X_avg, train_X_avg.index.to_series())
# add siesta 
train_X_avg["siesta"] = train_X_avg.hour.apply(siesta)

#- average weighted 6 x N Done
train_avg_X_weighted  = weighted_avg_ds(X_train, weight_col=False, weight_inplace=False, weights_dict={"Madrid":1.5/6,"Barcelona":1.5/6,"Valencia":1/6,"Seville":1/6,"Zaragoza":0.5/6,"Malaga":0.5/6 })
#transforming index to time features
train_avg_X_weighted = make_time_features(train_avg_X_weighted, train_avg_X_weighted.index.to_series())
# add siesta 
train_avg_X_weighted["siesta"] = train_avg_X_weighted.hour.apply(siesta)

# - 6x6xN weighted  
train_X_weighted_std = weighted_avg_ds(X_train, weight_col=False, weight_inplace=True, weights_dict={"Madrid":1.5/6,"Barcelona":1.5/6,"Valencia":1/6,"Seville":1/6,"Zaragoza":0.5/6,"Malaga":0.5/6 })
# transforming index to time features
train_X_weighted_std = make_time_features(train_X_weighted_std, train_X_weighted_std.index.to_series())
 # add siesta 
train_X_weighted_std["siesta"] = train_X_weighted_std.hour.apply(siesta)

 #- 6x7xX weight as a separate col for each city 
train_X_weighted_col = weighted_avg_ds(X_train, weight_col=True, weight_inplace=False, weights_dict={"Madrid":1/6,"Barcelona":1/6,"Valencia":1/6,"Seville":1/6,"Zaragoza":1/6,"Malaga":1/6 })
# transforming index to time features
train_X_weighted_col = make_time_features(train_X_weighted_col, train_X_weighted_col.index.to_series())
# add siesta 
train_X_weighted_col["siesta"] = train_X_weighted_col.hour.apply(siesta)
