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

# Import the user-defined modules
import sys
sys.path.append(path+'/../src')
import preprocess_ts, conduct_eda, prepare_for_rwcv, build_models
from preprocess_ts import *
from conduct_eda import *
from prepare_for_rwcv import *
from build_models import *

# Execute rwcv
def rwcv(dataset, model, DV_shifts, IV_lags=2, n_train_days=480, n_valid_days=60, n_test_days=60, metric=f1_score, feature_selection=True, feature_importances=False):
    """
    Iterate prediction for the given data as train for n_train_days days, validation for n_valid_days days, and test for n_test_days days

    Params:
        df: pd.DataFrame (Multivariate Time Series data longer than n_train_days+n_valid_days+n_test_days)
        model: str ({"Logistic", "RFC", "LGBMC", "FNN", })
        modelling: (Choose which algorithm to be used)
        DV_colname: str (The column name of the dependent variable focused in the experiment)
        DV_shifts: int (The number of differences btw DV and IVs)
        IV_lags: int (How many lags of IVs should be added as IVs)
        n_train_days: int
        n_valid_days: int
        n_test_days: int (This is also the same as the spacing between cutoff dates)
        metric: (Choose which metric to be used from {accuracy_score, precision_score, recall_score, f1_score})
        feature_selection: boolean (if true, BorutaPy choose variables)
        feature_importances: boolean (If True, feature imporances are given as a dataframe)

    Returns:
        cv_results: pd.DataFrame (the table of experimental results)

    """
    # Set up the dependent variable to analyze
    DV_colname="BTC_Price_LogReturn_binary"
    
    # Import the dataset
    blockchain_altcoin_macro = pd.read_pickle(path+"/03_primary/blockchain_altcoin_macro.pickle")
    blockchain_altcoin = pd.read_pickle(path+"/03_primary/blockchain_altcoin.pickle")
    blockchain_macro = pd.read_pickle(path+"/03_primary/blockchain_macro.pickle")
    blockchain = pd.read_pickle(path+"/03_primary/blockchain.pickle")
    
    # Choose the dataset to use for modelling
    if dataset == "BlockchainAltcoinMacro":
        df = blockchain_altcoin_macro.copy()
    elif dataset == "BlockchainAltcoin":
        df = blockchain_altcoin.copy()
    elif dataset == "BlockchainMacro":
        df = blockchain_macro.copy()
    elif dataset == "Blockchain":
        df = blockchain.copy()
    else:
        pass
    
    # Choose the model to build
    if model == "Logistic":
        modelling = Logistic
    elif model == "LGBMC":
        modelling = LGBMC
    elif model == "RFC":
        modelling = RFC
    elif model == "FNN":
        modelling = FNN
    else:
        pass
    
    # Generate dependent variable for the whole data range
    df["BTC_Price_LogReturn"] = np.log1p(df["BTC_Price"]).diff(periods=DV_shifts)
    df["BTC_Price_LogReturn_binary"] = (df["BTC_Price_LogReturn"] > 0).astype(int)
    df = df.dropna()

    # Set the initial day as the starting day of training and split the dataset by using the "split_ts" function
    train_start = df.index[0]
    train, valid, test, test_end = split_ts(df, train_start, n_train_days=n_train_days, n_valid_days=n_valid_days, n_test_days=n_test_days)

    # Best_params
    list_best_params = []
    
    # Independent variables used in each CV 
    independent_variables = []

    # Pred Series
    ModelPred_series = pd.DataFrame([])
    RandomPred_series = pd.DataFrame([])
    NaivePred_series = pd.DataFrame([])
    
    # Accuracy Scores
      # Forecasting Model
    train_valid_ModelScores = []  # Create charts
    test_ModelScores = []  # Create charts
    train_valid_ModelAccuracies = []
    test_ModelAccuracies = []
    train_valid_ModelPrecisions = []
    test_ModelPrecisions = []
    train_valid_ModelRecalls = []
    test_ModelRecalls = []
    train_valid_ModelF1s = []
    test_ModelF1s = []
      # Baseline Model (Random guess)
    train_valid_RandomScores = []  # Create charts
    test_RandomScores = []  # Create charts
    train_valid_RandomAccuracies = []
    test_RandomAccuracies = []
    train_valid_RandomPrecisions = []
    test_RandomPrecisions = []
    train_valid_RandomRecalls = []
    test_RandomRecalls = []
    train_valid_RandomF1s = []
    test_RandomF1s = []
      # Naive Forecasting based on Random Walk Hypothesis
    train_valid_NaiveScores = []  # Create charts
    test_NaiveScores = []  # Create charts
    train_valid_NaiveAccuracies = []
    test_NaiveAccuracies = []
    train_valid_NaivePrecisions = []
    test_NaivePrecisions = []
    train_valid_NaiveRecalls = []
    test_NaiveRecalls = []
    train_valid_NaiveF1s = []
    test_NaiveF1s = []
    
    # Simulated Returns
    train_valid_AlgoReturns = []
    test_AlgoReturns = []
    train_valid_RandomReturns = []
    test_RandomReturns = []
    train_valid_NaiveReturns = []
    test_NaiveReturns = []
    train_valid_BuyAndHolds = []
    test_BuyAndHolds = []

    # Feature Importances
    importance_initial = pd.DataFrame([])
    
    # The number of CV and time ranges
    counts = []
    test_starts = []
    test_ends = []
    
    # Execution time
    list_execution_cv= []
    
    # Start iteration here
    count = 1
    start_time = time.perf_counter()
    while test_end < df.index[-1]:
        # Count the number of CVs and executed time
        print("CV: ", count)
        start_cv = time.perf_counter()
        
        # Split the data to train, valid and test
        train, valid, test, test_end = split_ts(df, train_start, n_train_days=n_train_days, n_valid_days=n_valid_days, n_test_days=n_test_days)
        
        # Store train_start and test_end
        test_starts.append(test.index[0])
        test_ends.append(test_end-datetime.timedelta(days=1))
        
        # Build the model
        optimized_model, best_params, full_preprocessed, X_train_valid, y_train_valid, X_test_preprocessed, y_test_preprocessed, ModelPred_train_valid, ModelPred_test = \
            modelling(train, valid, test, DV_colname, DV_shifts, IV_lags, metric, feature_selection)

        # Hyperparams
        list_best_params.append(best_params)
        
        # Independent variables used in the model
        independent_variables.append(X_train_valid.columns.to_list())
        
        # Model prediction
        ModelPred_all = pd.concat([pd.Series(ModelPred_train_valid), pd.Series(ModelPred_test)])
        ModelPred_all.index = full_preprocessed.index
        ModelPred_series = pd.concat([ModelPred_series, ModelPred_all[test.index[0]:]])
    
        # Random prediction
        RandomPred_train_valid = np.round(np.random.rand(len(y_train_valid)))
        RandomPred_test = np.round(np.random.rand(len(y_test_preprocessed)))
        RandomPred_all = pd.concat([pd.Series(RandomPred_train_valid), pd.Series(RandomPred_test)])
        RandomPred_all.index = full_preprocessed.index
        RandomPred_series = pd.concat([RandomPred_series, RandomPred_all[test.index[0]:]])
        
        # Naive prediction
        NaivePred_train_valid = (full_preprocessed[y_train_valid.index[0]: y_train_valid.index[-1]]["BTC_Price_LogReturn"] > 0).astype(int)
        NaivePred_test = (full_preprocessed[y_test_preprocessed.index[0]: y_test_preprocessed.index[-1]]["BTC_Price_LogReturn"] > 0).astype(int)
        NaivePred_all = pd.concat([pd.Series(NaivePred_train_valid), pd.Series(NaivePred_test)])
        NaivePred_all.index = full_preprocessed.index
        NaivePred_series = pd.concat([NaivePred_series, NaivePred_all[test.index[0]:]])
    
        # Accuracy scores
          # The accuracy scores of forecasting model
        train_valid_ModelScore, test_ModelScore = metric(y_train_valid, ModelPred_train_valid), metric(y_test_preprocessed, ModelPred_test)
        train_valid_ModelAccuracy, test_ModelAccuracy = accuracy_score(y_train_valid, ModelPred_train_valid), accuracy_score(y_test_preprocessed, ModelPred_test)
        train_valid_ModelPrecision, test_ModelPrecision = precision_score(y_train_valid, ModelPred_train_valid), precision_score(y_test_preprocessed, ModelPred_test)
        train_valid_ModelRecall, test_ModelRecall = recall_score(y_train_valid, ModelPred_train_valid), recall_score(y_test_preprocessed, ModelPred_test)
        train_valid_ModelF1, test_ModelF1 = f1_score(y_train_valid, ModelPred_train_valid), f1_score(y_test_preprocessed, ModelPred_test)
          # The accuracy scores of random guess
        train_valid_RandomScore, test_RandomScore = metric(y_train_valid, RandomPred_train_valid), metric(y_test_preprocessed, RandomPred_test)
        train_valid_RandomAccuracy, test_RandomAccuracy = accuracy_score(y_train_valid, RandomPred_train_valid), accuracy_score(y_test_preprocessed, RandomPred_test)
        train_valid_RandomPrecision, test_RandomPrecision = precision_score(y_train_valid, RandomPred_train_valid), precision_score(y_test_preprocessed, RandomPred_test)
        train_valid_RandomRecall, test_RandomRecall = recall_score(y_train_valid, RandomPred_train_valid), recall_score(y_test_preprocessed, RandomPred_test)
        train_valid_RandomF1, test_RandomF1 = f1_score(y_train_valid, RandomPred_train_valid), f1_score(y_test_preprocessed, RandomPred_test)
          # The accuracy scores of naive forecasting
        train_valid_NaiveScore, test_NaiveScore = metric(y_train_valid, NaivePred_train_valid), metric(y_test_preprocessed, NaivePred_test)
        train_valid_NaiveAccuracy, test_NaiveAccuracy = accuracy_score(y_train_valid, NaivePred_train_valid), accuracy_score(y_test_preprocessed, NaivePred_test)
        train_valid_NaivePrecision, test_NaivePrecision = precision_score(y_train_valid, NaivePred_train_valid), precision_score(y_test_preprocessed, NaivePred_test)
        train_valid_NaiveRecall, test_NaiveRecall = recall_score(y_train_valid, NaivePred_train_valid), recall_score(y_test_preprocessed, NaivePred_test)
        train_valid_NaiveF1, test_NaiveF1 = f1_score(y_train_valid, NaivePred_train_valid), f1_score(y_test_preprocessed, NaivePred_test)

          # Store all the results to one list and create prediction series
            # Forecasting Model
        train_valid_ModelScores.append(train_valid_ModelScore)
        test_ModelScores.append(test_ModelScore)
        train_valid_ModelAccuracies.append(train_valid_ModelAccuracy)
        test_ModelAccuracies.append(test_ModelAccuracy)
        train_valid_ModelPrecisions.append(train_valid_ModelPrecision)
        test_ModelPrecisions.append(test_ModelPrecision)
        train_valid_ModelRecalls.append(train_valid_ModelRecall)
        test_ModelRecalls.append(test_ModelRecall)
        train_valid_ModelF1s.append(train_valid_ModelF1)
        test_ModelF1s.append(test_ModelF1)
            # Baseline model (Random guess)
        train_valid_RandomScores.append(train_valid_RandomScore)
        test_RandomScores.append(test_RandomScore)
        train_valid_RandomAccuracies.append(train_valid_RandomAccuracy)
        test_RandomAccuracies.append(test_RandomAccuracy)
        train_valid_RandomPrecisions.append(train_valid_RandomPrecision)
        test_RandomPrecisions.append(test_RandomPrecision)
        train_valid_RandomRecalls.append(train_valid_RandomRecall)
        test_RandomRecalls.append(test_RandomRecall)
        train_valid_RandomF1s.append(train_valid_RandomF1)
        test_RandomF1s.append(test_RandomF1)
            # Naive Forecasting
        train_valid_NaiveScores.append(train_valid_NaiveScore)
        test_NaiveScores.append(test_NaiveScore)
        train_valid_NaiveAccuracies.append(train_valid_NaiveAccuracy)
        test_NaiveAccuracies.append(test_NaiveAccuracy)
        train_valid_NaivePrecisions.append(train_valid_NaivePrecision)
        test_NaivePrecisions.append(test_NaivePrecision)
        train_valid_NaiveRecalls.append(train_valid_NaiveRecall)
        test_NaiveRecalls.append(test_NaiveRecall)
        train_valid_NaiveF1s.append(train_valid_NaiveF1)
        test_NaiveF1s.append(test_NaiveF1)
        
        # Algo Trading
          # Take positions depending on model forecasting 
        full_preprocessed["Position"] = 0
        full_preprocessed.loc[y_train_valid.index[0]: y_train_valid.index[-1], "Position"] = ModelPred_train_valid
        full_preprocessed.loc[y_test_preprocessed.index[0]:y_test_preprocessed.index[-1], "Position"] = ModelPred_test
        full_preprocessed["AlgoReturn"] = full_preprocessed["Position"] * df[full_preprocessed.index[0]: full_preprocessed.index[-1]]["BTC_Price_LogReturn"]
          # Take positions depending on random forecasting
        full_preprocessed["RandomPosition"] = 0
        full_preprocessed.loc[y_train_valid.index[0]: y_train_valid.index[-1], "RandomPosition"] = RandomPred_train_valid
        full_preprocessed.loc[y_test_preprocessed.index[0]:y_test_preprocessed.index[-1], "RandomPosition"] = RandomPred_test
        full_preprocessed["RandomReturn"] = full_preprocessed["RandomPosition"] * df[full_preprocessed.index[0]: full_preprocessed.index[-1]]["BTC_Price_LogReturn"]
          # Take positions depending on naive forecasting
        full_preprocessed["NaivePosition"] = 0
        full_preprocessed.loc[y_train_valid.index[0]: y_train_valid.index[-1], "NaivePosition"] = NaivePred_train_valid
        full_preprocessed.loc[y_test_preprocessed.index[0]:y_test_preprocessed.index[-1], "NaivePosition"] = NaivePred_test
        full_preprocessed["NaiveReturn"] = full_preprocessed["NaivePosition"] * df[full_preprocessed.index[0]: full_preprocessed.index[-1]]["BTC_Price_LogReturn"]
        
          # Calculate the sum of simulated Algo-Returns
        train_valid_AlgoReturns.append(full_preprocessed[y_train_valid.index[0]:y_train_valid.index[-1]]["AlgoReturn"].sum())
        test_AlgoReturns.append(full_preprocessed[y_test_preprocessed.index[0]: y_test_preprocessed.index[-1]]["AlgoReturn"].sum())
          # Calculate the sum of simulated random forecasting returns
        train_valid_RandomReturns.append(full_preprocessed[y_train_valid.index[0]:y_train_valid.index[-1]]["RandomReturn"].sum())
        test_RandomReturns.append(full_preprocessed[y_test_preprocessed.index[0]: y_test_preprocessed.index[-1]]["RandomReturn"].sum())
          # Calculate the sum of simulated Naive forecasting returns
        train_valid_NaiveReturns.append(full_preprocessed[y_train_valid.index[0]: y_train_valid.index[-1]]["NaiveReturn"].sum())
        test_NaiveReturns.append(full_preprocessed[y_test_preprocessed.index[0]: y_test_preprocessed.index[-1]]["NaiveReturn"].sum())
          # Calculate the sum of Buy-and-Hold returns
        train_valid_BuyAndHolds.append(full_preprocessed[y_train_valid.index[0]: y_train_valid.index[-1]]["BTC_Price_LogReturn"].sum())
        test_BuyAndHolds.append(full_preprocessed[y_test_preprocessed.index[0]: y_test_preprocessed.index[-1]]["BTC_Price_LogReturn"].sum())

        
        # Feature Importances
        if feature_importances:
            fi = pd.DataFrame(optimized_model.feature_importances_ / sum(optimized_model.feature_importances_), index=X_train_valid.columns, columns=["importance_"+str(count)])
            fi = pd.concat([importance_initial, fi], axis=1)
        else:
            pass

        # Record how many times loop was executed
        counts.append(count)
        count +=1
    
        
        # Tell me the executed time in each CV
        execution_cv = time.perf_counter() - start_cv
        print("Execution time per CV: ", execution_cv)
        list_execution_cv.append(execution_cv)
        
        # Add days for iteration of the process
        train_start += datetime.timedelta(days=n_test_days)
    
    # Tell me when it is finished
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    finished_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    print("Rolling Window Cross Validation has been finished: ", finished_time)
    print("Execution time (second) is: ", execution_time)

    # score metrics info
    if metric == accuracy_score:
        mtrc = "accuracy"
    elif metric == precision_score:
        mtrc = "precision"
    elif metric == recall_score:
        mtrc = "recall"
    elif metric == f1_score:
        mtrc = "f1"

    # Actual and Pred series
    ModelPred_series.columns = ["ModelPred_series"]
    RandomPred_series.columns = ["RandomPred_series"]
    NaivePred_series.columns = ["NaivePred_series"]
    Actual_series = df[ModelPred_series.index[0]: ModelPred_series.index[-1]][DV_colname]
    Actual_series.columns = ["Actual_series"]
    forecast_series = pd.concat([ModelPred_series, RandomPred_series, NaivePred_series, Actual_series, pd.DataFrame(df["BTC_Price_LogReturn"])], axis=1, join="inner")
    forecast_series["model"] = [model]*len(forecast_series)
    forecast_series["dataset"] = [dataset]*len(forecast_series)
    forecast_series["feature_selection"] = [feature_selection]*len(forecast_series)
    forecast_series["DV_shifts"] =  [DV_shifts]*len(forecast_series)
    forecast_series["IV_lags"] = [IV_lags]*len(forecast_series)
    forecast_series["metric"] = [mtrc]*len(forecast_series)

    
    # Output the results
      # Create a result table
    cv_results = pd.DataFrame({
                              # Basic Info
                              "CV": counts, "test_start": test_starts, "test_end": test_ends, "model": [model]*len(counts), 
                              "best_params": list_best_params,  "dataset": dataset,
                              # Hyper-Hyperparams
                              "feature_selection": [feature_selection]*len(counts), 
                              "dependent_variable": [DV_colname]*len(counts), "independent_variables": independent_variables,
                              "DV_shifts": [DV_shifts]*len(counts), "IV_lags": [IV_lags]*len(counts), "metric": [mtrc]*len(counts),
                              "n_train_days": [n_train_days]*len(counts), "n_valid_days": [n_valid_days]*len(counts), "n_test_days": [n_test_days]*len(counts),
                              # Forecasting Scores
                              "train_valid_model_f1": train_valid_ModelF1s, 
                              "test_model_f1": test_ModelF1s,                               
                              "train_valid_random_f1": train_valid_RandomF1s, 
                              "test_random_f1": test_RandomF1s,
                              "train_valid_naive_f1": train_valid_NaiveF1s, 
                              "test_naive_f1": test_NaiveF1s,
                              "train_valid_model_accuracy": train_valid_ModelAccuracies, 
                              "test_model_accuracy": test_ModelAccuracies,                               
                              "train_valid_random_accuracy": train_valid_RandomAccuracies, 
                              "test_random_accuracy": test_RandomAccuracies,
                              "train_valid_naive_accuracy": train_valid_NaiveAccuracies, 
                              "test_naive_accuracy": test_NaiveAccuracies,
                              # Simulated Returns
                              "train_valid_AlgoReturns": train_valid_AlgoReturns, "test_AlgoReturns": test_AlgoReturns,
                              "train_valid_RandomReturns": train_valid_RandomReturns, "test_RandomReturns": test_RandomReturns,
                              "train_valid_NaiveReturns": train_valid_NaiveReturns, "test_NaiveReturns": test_NaiveReturns,                              
                              "train_valid_BuyAndHolds": train_valid_BuyAndHolds, "test_BuyAndHolds": test_BuyAndHolds,
                              # Experimental Info
                              "execution_time": list_execution_cv, "finished_time": [finished_time]*len(counts)
                              }).set_index("CV")

    
    # Calculate the average feature importances of each variable
    if feature_importances:
        fi = pd.DataFrame(fi.T.describe().T["mean"].sort_values(ascending=False))
        fi = fi.rename({"mean": "average_value_of_feature_importances"}, axis=1)
        fi.to_csv(path+"/07_model_output/feature_importances/"+model+"_"+dataset+"_"+"FeatureSelection"+str(feature_selection)+"_"+"DVshifts"+str(DV_shifts)+"_"+"IVlags"+str(IV_lags)+"_"+mtrc+"_feature_importances.csv")
        print(fi.head(10))
    else:
        pass
    
    # Store the result table
      # Create the series file
    forecast_series.to_csv(path+"/07_model_output/actual_pred_series/"+model+"_"+dataset+"_"+"FeatureSelection"+str(feature_selection)+"_"+"DVshifts"+str(DV_shifts)+"_"+"IVlags"+str(IV_lags)+"_"+mtrc+"_results_series.csv")
      # Create the result file
    cv_results.to_csv(path+"/07_model_output/result_table/"+model+"_"+dataset+"_"+"FeatureSelection"+str(feature_selection)+"_"+"DVshifts"+str(DV_shifts)+"_"+"IVlags"+str(IV_lags)+"_"+mtrc+"_results_table.csv")
    

    
    return cv_results
