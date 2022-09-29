import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 205)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import describe

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.simplefilter('ignore')

# EDA
def describe_summary_statistics(df):
    """
    Add skewness and kurtosis to df.describe()
    
    Params:
        df: pd.DataFrame
    
    Return:
        df: pd.DataFrame (Descriptive Statistics table)
    
    """
    skewness = pd.DataFrame(data={df.columns[0]: describe(df[df.columns[0]].dropna())[4]}, index=["skewness"])
    kurtosis = pd.DataFrame(data={df.columns[0]: describe(df[df.columns[0]].dropna())[5]}, index=["kurtosis"])
    for col in df.columns[1:]:
        skewness_new = pd.DataFrame(data={col: describe(df[col].dropna())[4]}, index=["skewness"])
        skewness = pd.merge(skewness, skewness_new, left_index=True, right_index=True)
        kurtosis_new = pd.DataFrame(data={col: describe(df[col].dropna())[5]}, index=["kurtosis"])
        kurtosis = pd.merge(kurtosis, kurtosis_new, left_index=True, right_index=True)
    
    summary_stats = df.describe().append([skewness, kurtosis])
    
    return summary_stats
    
    
def viz_boxplot_hist_ts(df):
    """
    Visualize boxplot, histgram and time series on the all variables of a given dataframe
    
    Params:
        df: pd.DataFrame
        
    """
    for col in df.columns:
        fig = plt.figure(figsize=(10,8))
        ax1 = plt.subplot2grid((2,2), (0,0))
        ax2 = plt.subplot2grid((2,2), (0,1))
        ax3 = plt.subplot2grid((2,2), (1,0), colspan=2)
    
        ax1.boxplot(df[col])
        ax1.set_title("Boxplot of " + col)
        ax1.set_ylabel(col)
        ax1.grid(True)
    
        ax2.hist(df[col], bins=100)
        ax2.set_title("Histgram of " + col)
        ax2.set_xlabel(col)
        ax2.set_ylabel("Frequency")
        ax2.grid(True)
    
        ax3.plot(df[col])
        ax3.set_title("Time Series of " + col)
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Value")
        ax3.grid(True)
        
        fig.tight_layout()
        fig.show()
    
def viz_violin_hist_ts(df):
    """
    Visualize violinplot, histgram and time series on the all variables of a given dataframe
    
    Params:
        df: pd.DataFrame
        
    """
    for col in df.columns:
        fig = plt.figure(figsize=(10,8))
        ax1 = plt.subplot2grid((2,2), (0,0))
        ax2 = plt.subplot2grid((2,2), (0,1))
        ax3 = plt.subplot2grid((2,2), (1,0), colspan=2)
    
        sns.violinplot(x=df[col], ax=ax1)
        ax1.set_title("Violinplot of " + col)
        ax1.set_ylabel(col)
        ax1.grid(True)
    
        ax2.hist(df[col], bins=100)
        ax2.set_title("Histgram of " + col)
        ax2.set_xlabel(col)
        ax2.set_ylabel("Frequency")
        ax2.grid(True)
    
        ax3.plot(df[col])
        ax3.set_title("Time Series of " + col)
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Value")
        ax3.grid(True)
        
        fig.tight_layout()
        fig.show()


def viz_violin(df):
    """
    Plot violinplot for a given dataframe
    
    Params:
        df: pandas.DataFrame
    
    """
    
    for col in df.columns:
        sns.violinplot(x=df[col])
        plt.title("Violinplot: "+col)
        plt.xlabel(col)
        plt.ylabel("Value")
        plt.show()
        
def viz_boxplot(df):
    """
    Plot boxplot for a given dataframe
    
    Params:
        df: pandas.DataFrame
    
    """
    
    for col in df.columns:
        plt.boxplot(df[col])
        plt.title("Boxplot: "+col)
        plt.xlabel(col)
        plt.ylabel("Value")
        plt.show()
        


def viz_hist(df):
    """
    Plot histgram for a given dataframe
    
    Params:
        df: pandas.DataFrame
        
    """
    for col in df.columns:
        plt.hist(df[col], bins=100)
        plt.title("Histgram: " +col)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()

def viz_ts(df):
    """
    Plot time series of all variables in a given dataframe
    
    Params:
        df: pandas.DataFrame
        
    """
    
    for col in df.columns:
        plt.plot(df[col])
        plt.title(col)
        plt.xlabel("Time")
        plt.xticks(rotation=45)
        plt.ylabel("Value")
        plt.show()
        

# Time Series Analysis
def viz_pacf(df):
    """
    Visualize PACF plots of all variables in a given dataframe
    
    Params:
        df: pd.DataFrame
    
    """
    for col in df.columns:
        fig, ax = plt.subplots(figsize=(10,5))
        plot_pacf(df[col].dropna(), ax=ax, title="PACF of "+col);


def calculate_pacf(df, AC_thld=0.3):
    """
    Calculate PACF and display the autocorrelated orders on the variables in a given dataframe
    
    Params:
        df: pd.DataFrame
        AC_thld: how much autocorrelation coefficient is tested 
    
    """
    for col in df.columns:
        AC_orders = []
        for j in range(40):
            if (pacf(df[col])[j] > AC_thld) | ((pacf(df[col])[j] < -AC_thld)):
                AC_orders.append(j)
        print(col, AC_orders)
    
def test_adf(df, critical_value=0.05):
    adf_table = pd.DataFrame([])
    for col in df.columns:
        res = adfuller(df[col])
        adf_subtable = pd.DataFrame({"variables": [col], "test statistic": [res[0]], "p_value": [res[1]]})
        adf_table = pd.concat([adf_table, adf_subtable])
    return adf_table.set_index("variables")

def test_ljungbox(df, lags=10, critical_value=0.05):
    for col in df.columns:
        res = acorr_ljungbox(df[col].dropna(), lags=[lags])
        
        if res[1] < critical_value:
            print("Correlated: ", col)
        else:
            print("Not correlatad: ", col)
            

