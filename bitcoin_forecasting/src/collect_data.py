# Import the libraries
import pandas as pd
import numpy as np
from pandas_datareader import DataReader
import pandas_datareader.data as web
from datetime import datetime
import glob as glob
import os

# Google Drive
path = "..\data"


def yf_collect_data(dataset_name, dict_ticker, start, end, only_price_volume=True):
    # Make a folder to store the new dataset
    new_folder_raw = path+"/01_raw/"+dataset_name
    os.makedirs(new_folder_raw, exist_ok=True)

    # Store the keys and values of dict_ticker to lists
    list_keys = list(dict_ticker.keys())
    list_values = list(dict_ticker.values())

    # Get the data from yf and rename the columns to identify which dataframe corresponds to each ticker
    list_df = []
    for i in range(len(dict_ticker)): 
        list_df.append(pd.DataFrame(web.DataReader(list_values[i], "yahoo", start, end)))
        print(list_values[i])
        list_df[i].columns = [list_keys[i]+"_High", list_keys[i]+"_Low", list_keys[i]+"_Open", list_keys[i]+"_Close", list_keys[i]+"_Volume", list_keys[i]+"_AdjClose"]
        
        # Adjust datetime index
        list_df[i] = list_df[i].resample("D").mean()
        
        # Store the raw data to the folder of dataset_name in 01_raw
        list_df[i].to_csv(path+"/01_raw/"+dataset_name+"/"+list_keys[i]+".csv")

        # If conditions
        if only_price_volume:
            list_df[i] = list_df[i][[list_keys[i]+"_Volume", list_keys[i]+"_AdjClose"]]
        else:
            pass
    
    # Merge all the dataframes to the dataset
    df_dataset = list_df[0]
    for i in range(1, len(list_df)):
        df_dataset = pd.merge(df_dataset, list_df[i], left_index=True, right_index=True, how = "left")
        
    # Store the merged dataset to 01_raw
    df_dataset.to_pickle(path+"/01_raw/"+dataset_name+".pickle")
    
    return df_dataset