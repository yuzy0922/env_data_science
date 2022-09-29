# Import the libraries
import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error,r2_score


# Write own docstring memo
def time_series_split(y,train_start):
    
    """
    split the dataset into train data for 2 years, validation data for 2 months, test data for 2 months,
    when time series and the date to start training are given!

    Parameters:
    ----------
    y : Dataframe.Series

    train_start : The starting day for training
    
    Returns
    -------
    train : train data
    valid : validation data
    test : test data
    test_end : the last day for testing

    """

    # Variales 
    year2 = datetime.timedelta(days=365*2)
    month2 = datetime.timedelta(days=60)

    # Create train data
    train_end = train_start+year2
    train = y[(train_start <= y.index) & (y.index < train_end)]

    # Create validation data
    valid_start = train_end
    valid_end = valid_start + month2
    valid = y[(valid_start <= y.index) & (y.index < valid_end)]

    # Create test data
    test_start = valid_end
    test_end = test_start + month2
    test = y[(test_start <= y.index) & (y.index < test_end)]
    
    return train,valid,test,test_end

def output_result(answer,predict,metric=None):
    """
    Calculate the accuracy with the metrics which are given as a string like `RMSE`
    
    Parameters:
    -------
    answer : actual values
    predict : Predicted values
    metric : Accuracy Metrics（Choose from 'MAE','MSE','MSLE','r2_score','RMSE','RMSLE'）
    
    Return:
    -------
    Results
    
    """
    if metric == 'MAE':
        return mean_absolute_error(answer,predict)
    elif metric == 'MSE':
        return mean_squared_error(answer,predict)
    elif metric == 'MSLE':
        return mean_squared_log_error(answer,predict)
    elif metric == 'r2_score':
        return r2_score(answer,predict)
    elif metric == 'RMSE':
        return np.sqrt(mean_squared_error(answer,predict))
    elif metric == 'RMSLE':
        return np.sqrt(mean_squared_log_error(answer,predict))

def rolling_window_valid(df,modelling):
    
    """
    Iterate prediction for the given data as train for 2 years, validation for 2 months, test for 2 months (Multivariate)

    Parameters:
    ----------
    df : DataFrame.Series Multivariate Time Series data longer than 2 years + 4 months

    modelling : Choose the algorithm for modeling in the way of `modelling=` 
    
    Returns
    -------
    train_valid_results : Store the results of Train+Valid
    test_results : Store the results of test
    predicts : All of predicted results in the test period

    """
    
    #-------------The different points from rolling_window_valid-----------------
    train_start = df.index[0]
    train,valid,test,test_end = time_series_split(df,train_start)
    #---------------------------------------------------

    # Define the list to store the results in order to iterate the process
    train_valid_results = []
    test_results = []
    predicts = pd.DataFrame([])

    # Iteration starts here
    while test_end < df.index[-1]:

        train,valid,test,test_end = time_series_split(df,train_start)
        predict = modelling(train,valid,test)
        train_valid = pd.concat([train,valid])

        #-------------The different points from univariate rolling_window_valid-----------------
        # Calculate the results
        train_valid_result = output_result(train_valid.loc[predict.index[0]:,'01_market-price'],predict[:valid.index[-1]],metric='RMSLE')
        test_result = output_result(test['01_market-price'],predict[test.index[0]:],metric='RMSLE')
        #---------------------------------------------------
        
        # Store the results
        train_valid_results.append(train_valid_result)
        test_results.append(test_result)
        predicts = pd.concat([predicts,predict[test.index[0]:]])
        
        # Add 2 months for iteration of the process
        train_start += datetime.timedelta(days=60)
        
    return train_valid_results,test_results,predicts