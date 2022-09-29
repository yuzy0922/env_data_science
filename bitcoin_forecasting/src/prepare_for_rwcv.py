import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import seaborn as sns
from boruta import BorutaPy
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error,r2_score, accuracy_score, precision_score, recall_score, f1_score 
from sklearn.ensemble import *
import time
import datetime
import os

path = "..\data"


def split_ts(df, train_start, n_train_days, n_valid_days, n_test_days):
    """
    Split the dataset into train data for n_train_days days, validation data for n_valid_days days, test data for n_test_days days
    when time series and the starting day are given. 
    
    Params
        df: pd.DataFrame
        train_start: pandas._libs.tslibs.timestamps.Timestamp
        n_train_days: int
        n_valid_days: int
        n_test_days: int
    
    Returns
        train: pd.DataFrame (train data)
        valid: pd.DataFrame (validation data)
        test: pd.DataFrame (test data)
        test_end: datetime.index (the final day for testing)

    """

    # Variables
    dt_train_days = datetime.timedelta(days=n_train_days)
    dt_valid_days = datetime.timedelta(days=n_valid_days)
    dt_test_days = datetime.timedelta(days=n_test_days)

    # Create train data
    train_end = train_start + dt_train_days
    train = df[(train_start <= df.index) & (df.index < train_end)]

    # Create validation data
    valid_start = train_end
    valid_end = valid_start + dt_valid_days
    valid = df[(valid_start <= df.index) & (df.index < valid_end)]

    # Create test data
    test_start = valid_end
    test_end = test_start + dt_test_days
    test = df[(test_start <= df.index) & (df.index < test_end)]

    return train, valid, test, test_end


def preprocess_for_modelling(train, valid, test, DV_colname, DV_shifts, IV_lags, feature_selection):
    """
    Preprocess for modelling
    
    Params:
        train, valid, test: pd.DataFrame (Splitted data)
        DV_colname: str (The name of dependent variable used here)
        DV_shifts: int (The number of lags btw IVs and DV)
        IV_lags: int (How many lags of IVs should be added as IVs)
        feature_selection: boolean (if true, BorutaPy choose variables.)
    
    Returns:
        full_preprocessed, X_train_preprocessed, y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed, X_train_valid, y_train_valid: splitted and preprocessed data
    
    """

    # Concat the data again to conduct preprocessing
    full = pd.concat([train, valid, test])

    # Make BTC_Price_LogReturn binary
    full = full.drop(["BTC_Price"], axis=1)

    # Shift IVs by 1 or 2 
    full[full.columns[~full.columns.isin([DV_colname])]] = full.drop(DV_colname,axis=1).shift(DV_shifts)
    full_preprocessed = full[DV_shifts:]

    # Add the lags of IVs
    if IV_lags > 0:
        for col in full.drop(DV_colname, axis=1).columns:
            for lag in range(1, IV_lags+1):
                full[col + "_lag"+str(lag)] = full[col].shift(lag)
        full_preprocessed = full.dropna()
    else:
        pass

    # Split the data again because preprocessing for modelling was finished!
    train_preprocessed = full_preprocessed[:train.index[-1]]
    valid_preprocessed = full_preprocessed[valid.index[0]: valid.index[-1]]
    test_preprocessed = full_preprocessed[test.index[0]: test.index[-1]]

    # Split the data into X and y
    X_train_preprocessed = train_preprocessed.drop(DV_colname, axis=1)
    y_train_preprocessed = train_preprocessed[DV_colname]
    X_valid_preprocessed = valid_preprocessed.drop(DV_colname, axis=1)
    y_valid_preprocessed = valid_preprocessed[DV_colname]
    X_test_preprocessed = test_preprocessed.drop(DV_colname, axis=1)
    y_test_preprocessed = test_preprocessed[DV_colname]
    
    # Concat train and valid
    X_train_valid = pd.concat([X_train_preprocessed, X_valid_preprocessed])
    y_train_valid = pd.concat([y_train_preprocessed, y_valid_preprocessed])

    # StandardScaling inputs
      # X_train --> X_valid
    StdSc_train = StandardScaler()
    X_train_preprocessed = pd.DataFrame(StdSc_train.fit_transform(X_train_preprocessed),columns = X_train_preprocessed.columns)
    X_valid_preprocessed = pd.DataFrame(StdSc_train.transform(X_valid_preprocessed),columns = X_valid_preprocessed.columns)
      # X_train_valid --> X_test
    StdSc_train_valid = StandardScaler()
    X_train_valid = pd.DataFrame(StdSc_train_valid.fit_transform(X_train_valid),columns = X_train_valid.columns)
    X_test_preprocessed = pd.DataFrame(StdSc_train_valid.transform(X_test_preprocessed),columns = X_test_preprocessed.columns)
    
    # Features Selection by Boruta
    if feature_selection:    
        RFC = RandomForestClassifier(max_depth=7, random_state=201909)
        features_selector = BorutaPy(RFC, n_estimators='auto', two_step=False,verbose=2, random_state=201909)
        features_selector.fit(X_train_preprocessed.values,y_train_preprocessed.values)
        X_train_preprocessed = X_train_preprocessed.iloc[:,features_selector.support_]
        X_valid_preprocessed = X_valid_preprocessed.iloc[:,features_selector.support_]
        X_train_valid = X_train_valid.iloc[:,features_selector.support_]
        X_test_preprocessed = X_test_preprocessed.iloc[:,features_selector.support_]

    else:
        pass
    
    return full_preprocessed, X_train_preprocessed, y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed, X_train_valid, y_train_valid