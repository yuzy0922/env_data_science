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
from statsmodels.tsa.stattools import pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import ast
import itertools
import warnings
warnings.simplefilter('ignore')

path = "..\data"

# Important features selected by Boruta
def output_boruta(LogReturn=False):
    """
    Return the list of the variables chosen by Boruta.
    """
    
    imp_features_EndOfDay = pd.read_csv(path+"/08_reporting/table_selected_features_1DaysAhead.csv")
    imp_features_2DaysAhead = pd.read_csv(path+"/08_reporting/table_selected_features_2DaysAhead.csv")
    imp_features_3DaysAhead = pd.read_csv(path+"/08_reporting/table_selected_features_3DaysAhead.csv")
    imp_features = pd.concat([imp_features_EndOfDay, pd.concat([imp_features_2DaysAhead, imp_features_3DaysAhead])])
    imp_features = imp_features.reset_index(drop=True)
    v = []
    for i in [0,1,2]:
        v.append(ast.literal_eval(imp_features["selected_features"][i]))
    v_new = list(itertools.chain.from_iterable(v))

    v_removed = sorted(list(dict.fromkeys(v_new)))

    important_features = []
    for i in v_removed:
        if "lag1" in i:
            i = i.replace("_lag1", "")
        if "lag2" in i:
            i = i.replace("_lag2", "")
        i = i.replace("_LogReturn", "")
        important_features.append(i)
    for i in range(len(important_features)):
        if important_features[i] == "BTC":
            important_features[i] = "BTC_Price"
    important_features = list(dict.fromkeys(important_features))
    
    important_features_LogReturn = []
    for i in important_features:
        if i == "BTC_Price":
            i = "BTC_Price_LogReturn"
            important_features_LogReturn.append(i)
        elif (i == "blocks-size")|(i == "n-transactions-total")|(i == "total-bitcoins"):
            i = i + "_diff_LogReturn"
            important_features_LogReturn.append(i)
        else:
            important_features_LogReturn.append(i+"_LogReturn")
    if LogReturn:
        output = important_features_LogReturn
    else:
        output = important_features
    
    return output
    
def create_boruta_table():
    dataframe = pd.DataFrame(columns=["Variables", "End of Day", "Two days ahead", "Three days ahead"])
    dataframe["Variables"] = output_boruta(LogReturn=True)
    dataframe = dataframe.set_index("Variables")

    imp_features_1d = pd.read_csv(path+"/08_reporting/table_selected_features_1DaysAhead.csv")
    imp_features_1d = imp_features_1d.reset_index(drop=True)
    v_1d = []
    v_1d.append(ast.literal_eval(imp_features_1d["selected_features"][0]))
    v_new_1d = list(itertools.chain.from_iterable(v_1d))
    v_removed_1d = sorted(list(dict.fromkeys(v_new_1d)))

    imp_features_2d = pd.read_csv(path+"/08_reporting/table_selected_features_2DaysAhead.csv")
    imp_features_2d = imp_features_2d.reset_index(drop=True)
    v_2d = []
    v_2d.append(ast.literal_eval(imp_features_2d["selected_features"][0]))
    v_new_2d = list(itertools.chain.from_iterable(v_2d))
    v_removed_2d = sorted(list(dict.fromkeys(v_new_2d)))

    imp_features_3d = pd.read_csv(path+"/08_reporting/table_selected_features_3DaysAhead.csv")
    imp_features_3d = imp_features_3d.reset_index(drop=True)
    v_3d = []
    v_3d.append(ast.literal_eval(imp_features_3d["selected_features"][0]))
    v_new_3d = list(itertools.chain.from_iterable(v_3d))
    v_removed_3d = sorted(list(dict.fromkeys(v_new_3d)))

    dict_nth = {"End of Day": v_removed_1d, "Two days ahead": v_removed_2d, "Three days ahead": v_removed_3d}
    for nth in ["End of Day", "Two days ahead", "Three days ahead"]:
        for i in dataframe.index:
                if (i in dict_nth[nth]) & (i+"_lag1" not in dict_nth[nth]) & (i+"_lag2" not in dict_nth[nth]):
                    dataframe.loc[i, nth] = "0"
                elif (i not in dict_nth[nth]) & (i+"_lag1" in dict_nth[nth]) & (i+"_lag2" not in dict_nth[nth]):
                    dataframe.loc[i, nth] = "1"
                elif (i not in dict_nth[nth]) & (i+"_lag1" not in dict_nth[nth]) & (i+"_lag2" in dict_nth[nth]):
                    dataframe.loc[i, nth] = "2"
                elif (i in dict_nth[nth]) & (i+"_lag1" in dict_nth[nth]) & (i+"_lag2" not in dict_nth[nth]):
                    dataframe.loc[i, nth] = "0, 1"
                elif (i in dict_nth[nth]) & (i+"_lag1" not in dict_nth[nth]) & (i+"_lag2" in dict_nth[nth]):
                    dataframe.loc[i, nth] = "0, 2"
                elif (i not in dict_nth[nth]) & (i+"_lag1" in dict_nth[nth]) & (i+"_lag2" in dict_nth[nth]):
                    dataframe.loc[i, nth] = "1, 2"
                elif (i in dict_nth[nth]) & (i+"_lag1" in dict_nth[nth]) & (i+"_lag2" in dict_nth[nth]):
                    dataframe.loc[i, nth] = "1, 2, 3"
    return dataframe

