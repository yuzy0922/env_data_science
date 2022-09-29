import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 205)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')


# Preprocessing
def change_varnames_price_volume(df):
    """
    Change the variable names in prices and volumes.
    
    Params
        df: pandas.DataFrame
    
    """
    for col in df.columns:
        if "AdjClose" in col:
            replaced = col.replace("AdjClose", "Price")
        if "Volume" in col:
            replaced = col.replace("Volume", "Volume")
        df = df.rename({col: replaced}, axis=1)
    return df
    
def check_data(df):
    """
    Display df.head(), df.tail(), df.info(), df.shape at the same time
    
    Params
        df: pandas.DataFrame
    
    """
    
    print(df.shape)    
    print(df.head(3))
    print(df.tail(3))
    print(df.info())


def check_missing_values(df):
    """
    Visualize the information related to missing values
    
    Params:
        df: pandas.DataFrame
    
    """
    print(df.isnull().sum())
    print(sns.heatmap(df.isnull(), cbar=False))


def impute_missing_values(df, method="fillna_ffill", replace_inf=True):
    """
    Impute missing values with df.fillna() or df.interpolate()
    
    Params:
        df: pd.DataFrame
        method: str (Choose which kind of method to be used from "fillna_both", "fillna_ffill", "fillna_bfill", "interpolate_linear", "interpolate_spline2")
        replace_inf: boolean (True: replace np.inf and -np.inf with np.nan, False: do not replace them)
        
    """
    # Deal with np.inf and -np.inf cases
    df_bool_ninf = (df==-np.inf)
    df_bool_pinf = (df== np.inf)
    print("The number of -np.inf", df_bool_ninf.sum().sum())
    print("The number of np.inf", df_bool_pinf.sum().sum())
    # Replace np.inf, -np.inf with np.nan
    if replace_inf:
        df = df.replace([np.inf, -np.inf], np.nan)
    else:
        pass
    
    # Check the starus before and after imputation
    print("Before imputation: ", df.isnull().sum().sum())
    if method == "fillna_both":
        df = df.fillna(method="ffill")
        df = df.fillna(method="bfill")
    elif method == "fillna_ffill":
        df = df.fillna(method="ffill")
        if (df.loc[:df.index[0]].isnull().sum().sum() > 0) & (df.loc[df.index[1]:].isnull().sum().sum() == 0):
            df = df.drop(df.index[0])
        else:
            pass
    elif method == "interpolate_linear":
        df = df.interpolate(method="linear")
        if (df.loc[:df.index[0]].isnull().sum().sum() > 0) & (df.loc[df.index[1]:].isnull().sum().sum() == 0):
            df = df.drop(df.index[0])
        else:
            pass
    elif method == "interpolate_spline2":
        df = df.interpolate(method="spline", order=2)
        if (df.loc[:df.index[0]].isnull().sum().sum() > 0) & (df.loc[df.index[1]:].isnull().sum().sum() == 0):
            df = df.drop(df.index[0])
        else:
            pass
    else:
        pass
    print("After imputation: ", df.isnull().sum().sum())
    
    return df
    
def drop_zero_cols(df):
    """
    Drop the columns having all 0 values
    
    Params:
        df: pd.DataFrame
    """
    
    df = df.loc[:, (df != 0).any(axis=0)]
    
    return df
    
    
def take_log_return(df, drop_original=True):
    """
    Take LogReturn of all the variables in a given dataframe 
    
    Params:
        df: pd.DataFrame
        drop_original: boolean (True: LogReturn, False: original + LogReturn)

    Return:
        df: pd.DataFrame
    """
    # Drop the columns that contain only zero values
    df = df.loc[:, (df != 0).any(axis=0)]
    for col in df.columns:
        df[col+"_LogReturn"] = np.log1p(df[col]).diff()

    if drop_original:
        df = df.filter(like="LogReturn")
        return df
    else:
        return df
        
def make_log_return_binary(df, drop_original=True):
    """
    Identify the columns which incluse logreturn values and convert it to binary

    Params:
        df: pd.DataFrame
        drop_original: boolean (True: drop LogReturn variables, False: keep LogReturn variables)
    
    Return:
        df

    """
    drop_cols = []
    for col in df.columns:
        if "_LogReturn" in col:
            df[col+"_binary"] = (df[col] > 0).astype(int)
            if "BTC" not in col:
                drop_cols.append(col)
        else:
            pass
    if drop_original:
        df = df.drop(drop_cols, axis=1)
        return df
    else:
        return df

    