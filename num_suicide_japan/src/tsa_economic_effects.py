import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 205)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import t as t_dist
from scipy.stats import describe

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

import warnings
warnings.simplefilter('ignore')

def viz_decomposed(df, model="additive", original=False, trend=False, seasonal=False, resid=False, TR=True):
    """
    Decompose all the time series included in the dataframe
    
    Params:
        df: pd.DataFrame
        model: str ("additive" or "multiplicative")
        original: boolean
        trend: boolean
        seasonal: boolean
        resid: boolean
        seasonal_adjusted: boolean
        
    """
    df_decomposed = seasonal_decompose(df)
    for i in range(len(df.columns)):
        plt.figure(figsize=(8,6))
        if original:
            plt.plot(df_decomposed.observed[df.columns[i]], label="Original")
        else:
            pass
        if trend:
            plt.plot(df_decomposed.trend[df.columns[i]], label="Trend")
        else:
            pass
        if seasonal:
            plt.plot(df_decomposed.seasonal[df.columns[i]], label="Seasonal")
        else:
            pass
        if resid:
            plt.plot(df_decomposed.resid[df.columns[i]], label="Residual")
        else:
            pass
        if TR:
            plt.plot(df_decomposed.trend[df.columns[i]] + df_decomposed.resid[df.columns[i]], label="Seasonal Adjusted")
        else:
            pass       
        plt.title("Decomposition of " + df.columns[i])
        plt.xlabel("Time")
        plt.xticks(rotation=45)
        plt.ylabel("Value")
        plt.legend()
        plt.show()