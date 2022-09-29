# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import validation
from validation import time_series_split, output_result, rolling_window_valid

def make_lgb_dataset(df):
    
    """
    Adjust the dataframe to LGBM model
    
    Return:
    ------
    X : Dataframe which only includes explanatory variables
    Dataset : the dataset for LGBM
    
    """
    
    X = df.drop(['01_market-price_log'],axis=1)
    y = df['01_market-price_log']
    
    return X,lgb.Dataset(X,y)

def lgb_modelling(train,valid,test):
    
    """
    Make a prediction for LGBM multivariate.

    
    Parameters:
    ------
    train,valid,test : The data divided by the function `rolling_window_valid_multi`
    
    Return:
    ------
    predict : All predicted results including train, valid, test periods
    
    """
    
    # Merge the data
    full = pd.concat([train,valid,test])

    # Log transformation
    log_vars = full.columns
    for i in range(0,len(log_vars)):
        full[log_vars[i]+"_log"] = np.log1p(full[log_vars[i]])
        del full[log_vars[i]]

    # Take lags
    # The number of lags
    n = 5
    for i in range(1,n):
        full['log_lag'+str(i)] = full['01_market-price_log'].shift(i)
        
    # Calculate moving averages and standard errors
    rolling_n = [4,8,10,20]
    for i in rolling_n:
        full['01_market-price_log_rolling_mean_' + str(i)] = full['01_market-price_log'].rolling(i).mean().shift()
        full['01_market-price_log_rolling_std_' + str(i)] = full['01_market-price_log'].rolling(i).std().shift()
    # Calculate the returns
    full['log_return'] = full['01_market-price_log'].diff().shift()
    
    # Shift by n as to the variables which are unknown in the present
    n = 3
    full[full.columns[~full.columns.isin(['01_market-price_log'])]] = \
        full.drop('01_market-price_log',axis=1).shift(n)

    # Split the data again because preprocessing was finished.
    pp_train = full[:train.index[-1]]
    pp_valid = full[valid.index[0]:valid.index[-1]]
    pp_test = full[test.index[0]:test.index[-1]]
    
    # Create LGBM dataset
    X_pp_train, lgb_pp_train = make_lgb_dataset(pp_train)
    X_pp_valid, lgb_pp_valid = make_lgb_dataset(pp_valid)
    X_pp_test, lgb_pp_test = make_lgb_dataset(pp_test)
    
    lgb_params = {
            'objective': "regression",
            'metric':"rmse",
            'seed':2020
    }
    bst = lgb.train(
                lgb_params,
                train_set=lgb_pp_train,
                valid_sets=lgb_pp_valid,
                num_boost_round=1000,
                early_stopping_rounds=10,
                verbose_eval=False # Do not show inside processes
            )
    
    # Merge train with valid 
    pp_train_valid = pd.concat([pp_train,pp_valid])
    _,lgb_pp_train_valid = make_lgb_dataset(pp_train_valid)

    # Execute learning again
    best_bst = lgb.train(params=lgb_params,
                         train_set=lgb_pp_train_valid,
                         num_boost_round=bst.best_iteration)
    
    # Merge the all of explanatory variables in train, valid and test periods
    X_full = pd.concat([X_pp_train,X_pp_valid,X_pp_test])

    # Make a prediction
    predict = best_bst.predict(X_full)

    # Change the index of `predict`
    predict = pd.Series(predict)
    predict.index = list(train.index) + list(valid.index) + list(test.index)
    predict = np.exp(predict)

    # Plot the Time Series of answer, predict in valid and predict in test
    plt.figure(figsize=(8,4))
    plt.plot(full.loc[:test.index[-1],'01_market-price'])
    plt.plot(predict[:valid.index[-1]])
    plt.plot(predict[test.index[0]:])
    plt.legend(['answer','predict','predict_test'])
    plt.show()

    # Interpretation of accuracy metrics in valid and test periods
    train_valid_result = output_result(pd.concat([train,valid])['01_market-price'],predict[:valid.index[-1]],metric='RMSLE')
    test_result = output_result(test['01_market-price'],predict[test.index[0]:],metric='RMSLE')
    print('Train:',train_valid_result)
    print('Test:',test_result)
    
    return predict