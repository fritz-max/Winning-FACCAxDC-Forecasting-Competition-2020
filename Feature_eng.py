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
def siesta(hour_column,hours=[14,15,16]):
    siesta = hour_column[lambda x: 1 if x in hours else 0]
    return siesta
