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
def viz_pacf(df):
    """
    Visualize PACF plots of all variables in a given dataframe
    
    Params:
        df: pd.DataFrame
    
    """
    for i in range(len(df.columns)):
        fig, ax = plt.subplots(figsize=(10,5))
        plot_pacf(df[df.columns[i]].dropna(), ax=ax, title="PACF of "+df.columns[i]);


def calculate_pacf(df, AC_thld=0.3):
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

    