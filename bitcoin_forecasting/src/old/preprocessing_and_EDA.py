import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 205)
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import t as t_dist
from scipy.stats import describe

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import adfuller

import warnings
warnings.simplefilter('ignore')


# Preprocessing
def check_data(df):
    """
    Display df.head(), df.tail(), df.info(), df.shape at the same time
    
    Params
        df: pandas.DataFrame
    
    """
    
    display(df.shape)    
    display(df.head(3))
    display(df.tail(3))
    display(df.info())


def check_missing_values(df):
    """
    Visualize the information relate to missing values
    
    Params:
        df: pandas.DataFrame
    
    """
    display(df.isnull().sum())
    display(sns.heatmap(df.isnull(), cbar=False))


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
    
def drop_zeroCol(df):
    """
    Drop the columns having all 0 values
    
    Params:
        df: pd.DataFrame
    """
    
    df = df.loc[:, (df != 0).any(axis=0)]
    
    return df

def create_subset_df(df, sub_colname):
    """
    Create the subset of dataframe from variables having assigned substring

    Params: 
        df: pd.DataFrame 
        sub_colname: str (substring of variable names)
    
    Return:
        subset_df: pd.DataFrame (subset of original dataframe)
    """
    subset_cols = []
    for i in range(len(df.columns)):
        if sub_colname in df.columns[i]:
            subset_cols.append(df.columns[i])

    subset_df = df[subset_cols]
    
    return subset_df
    
    
def take_LogReturn(df, drop_original=True, p1=False):
    """
    Take LogReturn of all the variables in a given dataframe 
    
    Params:
        df: pd.DataFrame
        drop_original: boolean (True: LogReturn, False: original + LogReturn)
        1p: boolean (True: np.log1p(), False: np.log())
        
    Return:
        df: pd.DataFrame
    """
    # Drop the columns that contain only zeros
    df = df.loc[:, (df != 0).any(axis=0)]
    for i in range(len(df.columns)):
        if p1:
            df[df.columns[i]+"_LogReturn"] = np.log1p(df[df.columns[i]]).diff()
        else:
            df[df.columns[i]+"_LogReturn"] = np.log(df[df.columns[i]]+1e-10).diff() #+1e-10 avoids inf and -inf
    
    else:
        pass
        
    
    if drop_original:
        df = create_subset_df(df, "_LogReturn")
        return df
    else:
        return df
        
def make_LogReturn_binary(df, drop_original=True):
    """
    Identify the columns which incluse logreturn values and convert it to binary

    Params:
        df: pd.DataFrame
        drop_original: boolean (True: drop LogReturn variables, False: keep LogReturn variables)
    
    Return:
        df

    """
    drop_cols = []
    for i in range(len(df.columns)):
        if "_LogReturn" in df.columns[i]:
            df[df.columns[i]+"_binary"] = (df[df.columns[i]] > 0).astype(int)
            if "btc" not in df.columns[i]:
                drop_cols.append(df.columns[i])
        else:
            pass
    if drop_original:
        df = df.drop(drop_cols, axis=1)
        return df
    else:
        return df
    
    
def take_log(df, drop_original=True, p1=False):
    """
    Take log of all the variables in a given dataframe
    
    Params:
        df: pd.DataFrame
        drop_original: boolean (True: log, False: original + log)   
        p1: boolean (True: np.log1p(), False: np.log())
    
    Return:
        df: pd.DataFrame
    
    """
    
    # Drop the columns that contain only zeros
    df = df.loc[:, (df != 0).any(axis=0)]    
    for i in range(len(df.columns)):
        if p1:
            df[df.columns[i]+"_log"] = np.log1p(df[df.columns[i]])
        else:    
            df[df.columns[i]+"_log"] = np.log(df[df.columns[i]]+1e-10) ##+1e-10 avoids inf and -inf
            
    if drop_original:
        df = create_subset_df(df, "_log")
        return df
    else:
        return df

    

# EDA
def describe_summary_statistics(df):
    """
    Add skewness and kurtosis to df.describe()
    
    Params:
        df: pd.DataFrame
    
    Return:
        df: pd.DataFrame (Descriptive Statistics table)
    
    """
    skewness = pd.DataFrame(data={df.columns[0]: describe(df[df.columns[0]])[4]}, index=["skewness"])
    kurtosis = pd.DataFrame(data={df.columns[0]: describe(df[df.columns[0]])[5]}, index=["kurtosis"])
    for i in range(1, len(df.columns)):
        skewness_new = pd.DataFrame(data={df.columns[i]: describe(df[df.columns[i]])[4]}, index=["skewness"])
        skewness = pd.merge(skewness, skewness_new, left_index=True, right_index=True)
        kurtosis_new = pd.DataFrame(data={df.columns[i]: describe(df[df.columns[i]])[5]}, index=["kurtosis"])
        kurtosis = pd.merge(kurtosis, kurtosis_new, left_index=True, right_index=True)
    
    summary_stats = df.describe().append([skewness, kurtosis])
    
    return summary_stats
    
    
def viz_boxplot_hist_ts(df):
    """
    Visualize boxplot, histgram and time series on the all variables of a given dataframe
    
    Params:
        df: pd.DataFrame
        
    """
    for i in range(len(df.columns)):
        fig = plt.figure(figsize=(10,8))
        ax1 = plt.subplot2grid((2,2), (0,0))
        ax2 = plt.subplot2grid((2,2), (0,1))
        ax3 = plt.subplot2grid((2,2), (1,0), colspan=2)
    
        ax1.boxplot(df[df.columns[i]])
        ax1.set_title("Boxplot of " + df.columns[i])
        ax1.set_ylabel(df.columns[i])
        ax1.grid(True)
    
        ax2.hist(df[df.columns[i]], bins=100)
        ax2.set_title("Histgram of " + df.columns[i])
        ax2.set_xlabel(df.columns[i])
        ax2.set_ylabel("Frequency")
        ax2.grid(True)
    
        ax3.plot(df[df.columns[i]])
        ax3.set_title("Time Series of " + df.columns[i])
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Value")
        ax3.grid(True)
        
        fig.tight_layout()
        fig.show()
    

def viz_boxplot(df):
    """
    Plot boxplot for a given dataframe
    
    Params:
        df: pandas.DataFrame
    
    """
    
    for i in range(len(df.columns)):
        plt.boxplot(df[df.columns[i]])
        plt.title("Boxplot: "+df.columns[i])
        plt.xlabel(df.columns[i])
        plt.ylabel("Value")
        plt.show()


def viz_hist(df):
    """
    Plot histgram for a given dataframe
    
    Params:
        df: pandas.DataFrame
        
    """
    for i in range(len(df.columns)):
        plt.hist(df[df.columns[i]], bins=100)
        plt.title("Histgram: " +df.columns[i])
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()

def viz_ts(df):
    """
    Plot time series of all variables in a given dataframe
    
    Params:
        df: pandas.DataFrame
        
    """
    
    for i in range(0, len(df.columns)):
        plt.plot(df[df.columns.to_list()[i]])
        plt.title(df.columns.to_list()[i])
        plt.xlabel("Time")
        plt.xticks(rotation=45)
        plt.ylabel("Value")
        plt.show()
        

# Time Series Analysis
def viz_PACF(df):
    """
    Visualize PACF plots of all variables in a given dataframe
    
    Params:
        df: pd.DataFrame
    
    """
    for i in range(len(df.columns)):
        fig, ax = plt.subplots(figsize=(10,5))
        plot_pacf(df[df.columns[i]].dropna(), ax=ax, title="PACF of "+df.columns[i]);


def calculate_PACF(df, AC_thld=0.3):
    """
    Calculate PACF and display the autocorrelated orders on the variables in a given dataframe
    
    Params:
        df: pd.DataFrame
        AC_thld: how much autocorrelation coefficient is tested 
    
    """
    for i in range(len(df.columns)):
        AC_orders = []
        for j in range(40):
            if (pacf(df[df.columns[i]])[j] > AC_thld) | ((pacf(df[df.columns[i]])[j] < -AC_thld)):
                AC_orders.append(j)
        print(df.columns[i], AC_orders)
    
def test_adf(df):
    for i in range(len(df.columns)):
        res = adfuller(df[df.columns[i]])
        
        if res[1] < 0.05:
            print("Stationary: ", df.columns[i])
        else:
            print("Non-Stationary: ", df.columns[i])

    