import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error,r2_score, accuracy_score, precision_score, recall_score, f1_score 
import time
import datetime
import os

path = "/content/drive/My Drive/Master_Research/workspace/data"

# Import the user-defined modules
import sys
sys.path.append(path+'/../src')
import preprocessing_ts, conducting_EDA
from preprocessing_ts import *
from conducting_EDA import *


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
    
def preprocess_for_modelling(train, valid, test, DV_colname, DV_shifts, IV_lags):
    """
    Preprocess for modelling
    
    Params:
        train, valid, test: pd.DataFrame (Splitted data)
        DV_colname: str (The name of dependent variable used here)
        DV_shifts: int (The number of lags btw IVs and DV)
        IV_lags: int (How many lags of IVs should be added as IVs)
    
    Returns:
        full_preprocessed, X_train_preprocessed, y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed: splitted and preprocessed data
    
    """

    # Concat the data again to conduct preprocessing
    full = pd.concat([train, valid, test])

    # Make btc_LogReturn binary
    full = full.drop(["btc_price"], axis=1)

    # Shift the DV by n 
    full[full.columns[~full.columns.isin([DV_colname])]] = full.drop(DV_colname,axis=1).shift(DV_shifts)
    full_preprocessed = full[DV_shifts:]

    # Add the lags of IVs
    if IV_lags > 0:
        for i in range(len(full.drop(DV_colname, axis=1).columns)):
            for lag in range(1, IV_lags+1):
                full[full.drop(DV_colname, axis=1).columns[i] + "_lag"+str(lag)] = full[full.drop(DV_colname, axis=1).columns[i]].shift(lag)
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

    # StandardScaling inputs if IVs are binary values
        # X_train --> X_valid
    if full_preprocessed.max().max()!=1.0:
        StdSc_train = StandardScaler()
        X_train_preprocessed = pd.DataFrame(StdSc_train.fit_transform(X_train_preprocessed),columns = X_train_preprocessed.columns)
        X_valid_preprocessed = pd.DataFrame(StdSc_train.transform(X_valid_preprocessed),columns = X_valid_preprocessed.columns)
        # X_train_valid --> X_test
        StdSc_train_valid = StandardScaler()
        X_train_valid = pd.DataFrame(StdSc_train_valid.fit_transform(X_train_valid),columns = X_train_valid.columns)
        X_test_preprocessed = pd.DataFrame(StdSc_train_valid.transform(X_test_preprocessed),columns = X_test_preprocessed.columns)
    
    return full_preprocessed, X_train_preprocessed, y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed, X_train_valid, y_train_valid
    
def viz_results(df, DV_colname, ModelPred_series, y_test_preprocessed, test_ModelAccuracies, test_RandomAccuracies, test_AlgoReturns, test_BuyAndHolds, test_Randoms, fi, feature_importances, new_folder, model_name, finished_time):
    
    # Calculate the average forecasting scores
    print("Average of forecasting-model scores on test", np.mean(test_ModelAccuracies))
    print("Average of random-forecasting scores on test", np.mean(test_RandomAccuracies))

    # Calculate the average of total returns on each CV
    print("Average of total Algo return on test", np.mean(test_AlgoReturns))
    print("Average of total Buy-And-Hold return on test", np.mean(test_BuyAndHolds))
#    print("Average of total Random return on test", np.mean(test_Randoms))
   
    # Instantiate graphs
    fig = plt.figure(figsize=(12,10))
    ax1 = plt.subplot2grid((3,2), (0,0))
    ax2 = plt.subplot2grid((3,2), (0,1))
    ax3 = plt.subplot2grid((3,2), (1,0))
    ax4 = plt.subplot2grid((3,2), (1,1))
    ax5 = plt.subplot2grid((3,2), (2,0))
    ax6 = plt.subplot2grid((3,2), (2,1))

    # Visualize heatmap
    cm = metrics.confusion_matrix(df[ModelPred_series.index[0]: ModelPred_series.index[-1]][DV_colname], ModelPred_series["ModelPred_series"])
    df_cm = pd.DataFrame(cm, columns=np.unique(y_test_preprocessed), index = np.unique(y_test_preprocessed))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.heatmap(df_cm, annot=True, fmt='d', ax=ax1)
    ax1.set_title("Confusion Heatmap of the Forecasting Model")
    ax1.grid(False)
    
    # Visualize Classification Report
    target_names = list("01")
    clsf_report = metrics.classification_report(df[ModelPred_series.index[0]: ModelPred_series.index[-1]][DV_colname], ModelPred_series["ModelPred_series"], target_names=target_names, output_dict=True)
    print(metrics.classification_report(df[ModelPred_series.index[0]: ModelPred_series.index[-1]][DV_colname], ModelPred_series["ModelPred_series"], target_names=target_names, output_dict=False))
    clsf = pd.DataFrame(clsf_report).iloc[:-1, :]
    clsf = clsf.drop(["accuracy"], axis=1).T
    sns.heatmap(clsf, annot=True, ax=ax2)
    ax2.set_title("Classification Report")
    
    # Visualize the forecasting scores
    ax3.plot(test_ModelAccuracies, label="ModelPred-Accuracy")
    ax3.plot(test_RandomAccuracies, label="RandomPred-Accuracy")
    ax3.set_title("Forecasting Accuracies on each CV")
    ax3.set_xlabel("Times of Cross Validation")
    ax3.set_ylabel("Forecasting Accuracies")
    ax3.grid(True)
    ax3.legend()
 
    ax4.hist(test_ModelAccuracies, bins=6, label="ModelPred-Accuracy", alpha=0.3)
    ax4.hist(test_RandomAccuracies, bins=6, label="RandomPred-Accuracy", alpha=0.3)
    ax4.set_title("Distribution of Forecasting Accuracies on each CV")
    ax4.set_xlabel("Forecasting Scores")
    ax4.set_ylabel("Frequency")
    ax4.grid(True)
    ax4.legend()

    # Visualize the total returns
    ax5.plot(test_AlgoReturns, label="ModelPred Returns")
    ax5.plot(test_BuyAndHolds, label="Buy-and-Hold Returns")
#    ax5.plot(test_Randoms, label="RandomPred Returns")
    ax5.set_title("Total Returns on each CV")
    ax5.set_xlabel("Times of Cross Validation")
    ax5.set_ylabel("Total Returns")
    ax5.grid(True)
    ax5.legend()
    
    ax6.hist(test_AlgoReturns, label="ModelPred Returns", histtype='stepfilled', alpha=0.3)
    ax6.hist(test_BuyAndHolds, label="Buy-and-Hold Returns", histtype='stepfilled', alpha=0.3)
#    ax6.hist(test_Randoms, label="RandomPred Returns", histtype='stepfilled', alpha=0.3)
    ax6.set_title("Distribution of Total Returns on each CV")
    ax6.set_xlabel("Total Returns")
    ax6.set_ylabel("Frequency")
    ax6.grid(True)
    ax6.legend()
    
    fig.tight_layout()
    fig.show()
    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"all_figures.png", format="png", dpi=400)
    

def save_figs(df, DV_colname, ModelPred_series, y_test_preprocessed, test_ModelAccuracies, test_RandomAccuracies, test_AlgoReturns, test_BuyAndHolds, test_Randoms, fi, feature_importances, new_folder, model_name, finished_time):
    """
    Save the above axes individually.
    """
    # ax1
    plt.figure(figsize=(12,8))
    cm = metrics.confusion_matrix(df[ModelPred_series.index[0]: ModelPred_series.index[-1]][DV_colname], ModelPred_series["ModelPred_series"])
    df_cm = pd.DataFrame(cm, columns=np.unique(y_test_preprocessed), index = np.unique(y_test_preprocessed))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.title("Confusion Heatmap of the Forecasting Model")
    plt.grid(False)
    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"confusion_heatmap"+".png", format="png", dpi=400);
    
    # ax2
    plt.figure(figsize=(12,8))
    target_names = list("01")
    clsf_report = metrics.classification_report(df[ModelPred_series.index[0]: ModelPred_series.index[-1]][DV_colname], ModelPred_series["ModelPred_series"], target_names=target_names, output_dict=True)
    clsf = pd.DataFrame(clsf_report).iloc[:-1, :]
    clsf = clsf.drop(["accuracy"], axis=1).T
    sns.heatmap(clsf, annot=True)
    plt.title("Classification Report")
    plt.grid(False)
    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"classification_report"+".png", format="png", dpi=400);
    
    # ax3
    plt.figure(figsize=(12,8))
    plt.plot(test_ModelAccuracies, label="ModelPred-Accuracy")
    plt.plot(test_RandomAccuracies, label="RandomPred-Accuracy")
    plt.title("Forecasting Accuracies on each CV")
    plt.xlabel("Times of Cross Validation")
    plt.ylabel("Forecasting Accuracies")
    plt.grid(True)
    plt.legend()
    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"plot_accuracy"+".png", format="png", dpi=400);
    
    # ax4
    plt.figure(figsize=(12,8))
    plt.hist(test_ModelAccuracies, bins=6, label="ModelPred-Accuracies", alpha=0.3)
    plt.hist(test_RandomAccuracies, bins=6, label="RandomPred-Accuracies", alpha=0.3)
    plt.title("Distribution of Forecasting Accuracies on each CV")
    plt.xlabel("Forecasting Accuracies")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()
    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"dist_accuracy"+".png", format="png", dpi=400);
    
    # ax5
    plt.figure(figsize=(12,8))
    plt.plot(test_AlgoReturns, label="ModelPred Returns")
    plt.plot(test_BuyAndHolds, label="Buy-and-Hold Returns")
#    plt.plot(test_Randoms, label="RandomPred Returns")
    plt.title("Total Returns on each CV")
    plt.xlabel("Times of Cross Validation")
    plt.ylabel("Total Returns")
    plt.grid(True)
    plt.legend()
    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"plot_returns"+".png", format="png", dpi=400);
    
    # ax6
    plt.figure(figsize=(12,8))
    plt.hist(test_AlgoReturns, label="ModelPred Returns", histtype='stepfilled', alpha=0.3)
    plt.hist(test_BuyAndHolds, label="Buy-and-Hold Returns", histtype='stepfilled', alpha=0.3)
#    plt.hist(test_Randoms, label="RandomPred Returns", histtype='stepfilled', alpha=0.3)
    plt.title("Distribution of Total Returns on each CV")
    plt.xlabel("Total Returns")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()
    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"dist_returns"+".png", format="png", dpi=400);



# Execute rwcv_c
def rwcv_c(df, model_name, modelling, DV_colname="btc_LogReturn_binary", DV_shifts=1, IV_lags=2, n_train_days=600, n_valid_days=30, n_test_days=30, metric=f1_score, feature_importances=False):
    """
    Iterate prediction for the given data as train for n_train_days days, validation for n_valid_days days, and test for n_test_days days

    Params:
        df: pd.DataFrame (Multivariate Time Series data longer than n_train_days+n_valid_days+n_test_days)
        model_name: str ({"RFC", "LGBMC", "DNNC", "CNNC", "RNNC", "LSTMC", "BNNC", "attentionC"})
        modelling: (Choose which algorithm to be used)
        DV_colname: str (The column name of the dependent variable focused in the experiment)
        DV_shifts: int (The number of differences btw DV and IVs)
        IV_lags: int (How many lags of IVs should be added as IVs)
        n_train_days: int
        n_valid_days: int
        n_test_days: int (This is also the same as the spacing between cutoff dates)
        metric: (Choose which metric to be used from {accuracy_score, precision_score, recall_score, f1_score})
        feature_importances: boolean (If True, feature imporances are given as a dataframe)

    Returns:
        df_output: pd.DataFrame (the table of experimental results)

    """

    # Set the initial day as the starting day of training and split the dataset by using the "split_ts" function
    train_start = df.index[0]
    train, valid, test, test_end = split_ts(df, train_start, n_train_days=n_train_days, n_valid_days=n_valid_days, n_test_days=n_test_days)
   
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
    
    # Pred Series
    ModelPred_series = pd.DataFrame([])
    
    # Simulated Returns
    train_valid_AlgoReturns = []
    test_AlgoReturns = []
    train_valid_Randoms = []
    test_Randoms = []
    train_valid_BuyAndHolds = []
    test_BuyAndHolds = []
    
    # Best_params
    list_best_params = []
    
    # Feature Importances
    importance_initial = pd.DataFrame([])
    
    # The number of counts and the range
    counts = []
    test_starts = []
    test_ends = []
    
    # Time to execute
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
        
        # Build the model
        optimized_model, best_params, full_preprocessed, X_train_valid, y_train_valid, X_test_preprocessed, y_test_preprocessed = modelling(train, valid, test, DV_colname, DV_shifts, IV_lags, metric)
        
        # Predict y
          # Predict y on train_valid and test
        ModelPred_train_valid = np.round(optimized_model.predict(X_train_valid).flatten())
        ModelPred_test = np.round(optimized_model.predict(X_test_preprocessed).flatten())
          # Concat train_valid and test to create one series
        ModelPred_all = pd.concat([pd.Series(ModelPred_train_valid), pd.Series(ModelPred_test)])
        ModelPred_all.index = full_preprocessed.index
        ModelPred_series = pd.concat([ModelPred_series, ModelPred_all[test.index[0]:]])
        
        # Random prediction
        RandomPred_train_valid = np.round(np.random.rand(len(y_train_valid)))
        RandomPred_test = np.round(np.random.rand(len(y_test_preprocessed)))
        
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
        
        # Algo Trading
          # Take positions by the advanced model which forecasts whether returns are positive or negative
        full_preprocessed["Position"] = 0
        full_preprocessed.loc[y_train_valid.index[0]: y_train_valid.index[-1], "Position"] = ModelPred_train_valid
        full_preprocessed.loc[y_test_preprocessed.index[0]:y_test_preprocessed.index[-1], "Position"] = ModelPred_test
        full_preprocessed["AlgoReturn"] = full_preprocessed["Position"] * df[full_preprocessed.index[0]: full_preprocessed.index[-1]]["btc_LogReturn"]
          # Take positions by the random forecasting
        full_preprocessed["RandomPosition"] = 0
        full_preprocessed.loc[y_train_valid.index[0]: y_train_valid.index[-1], "RandomPosition"] = RandomPred_train_valid
        full_preprocessed.loc[y_test_preprocessed.index[0]:y_test_preprocessed.index[-1], "RandomPosition"] = RandomPred_test
        full_preprocessed["RandomReturn"] = full_preprocessed["RandomPosition"] * df[full_preprocessed.index[0]: full_preprocessed.index[-1]]["btc_LogReturn"]
         
          # Calculate the sum of Algo-Returns
        train_valid_AlgoReturns.append(full_preprocessed[y_train_valid.index[0]:y_train_valid.index[-1]]["AlgoReturn"].sum())
        test_AlgoReturns.append(full_preprocessed[y_test_preprocessed.index[0]: y_test_preprocessed.index[-1]]["AlgoReturn"].sum())
          # Calculate the sum of Buy-and-Hold returns
        train_valid_BuyAndHolds.append(df[y_train_valid.index[0]: y_train_valid.index[-1]]["btc_LogReturn"].sum())
        test_BuyAndHolds.append(df[y_test_preprocessed.index[0]: y_test_preprocessed.index[-1]]["btc_LogReturn"].sum())
          # Calculate the sum of random returns
        train_valid_Randoms.append(full_preprocessed[y_train_valid.index[0]:y_train_valid.index[-1]]["RandomReturn"].sum())
        test_Randoms.append(full_preprocessed[y_test_preprocessed.index[0]: y_test_preprocessed.index[-1]]["RandomReturn"].sum())
        
        # Hyperparams
        list_best_params.append(best_params)
        
        # Feature Importances
        if feature_importances:
            fi = pd.DataFrame(optimized_model.feature_importances_ / sum(optimized_model.feature_importances_), index=X_train_valid.columns, columns=["importance_"+str(count)])
            fi = pd.concat([importance_initial, fi], axis=1)
        else:
            pass

        # Record how many times loop was executed
        counts.append(count)
        count +=1
        
        # Store train_start and test_end
        test_starts.append(test.index[0])
        test_ends.append(test_end-datetime.timedelta(days=1))
        
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

    # Assign the colume name of ModelPred_series
    ModelPred_series.columns = ["ModelPred_series"]
    
    # Output the results
      # Create a result table
    df_output = pd.DataFrame({"CV": counts, "test_start": test_starts, "test_end": test_ends, "model": [model_name]*len(counts), "best_params": list_best_params,
                              # Forecasting Scores
                              "train_valid_model_f1": train_valid_ModelF1s, 
                              "test_model_f1": test_ModelF1s,                               
                              "train_valid_random_f1": train_valid_RandomF1s, 
                              "test_random_f1": test_RandomF1s,
                              # Profitability
                              "train_valid_AlgoReturns": train_valid_AlgoReturns, "test_AlgoReturns": test_AlgoReturns,
                              "train_valid_RandomReturns": train_valid_Randoms, "test_RandomReturns": test_Randoms,
                              "train_valid_BuyAndHolds": train_valid_BuyAndHolds, "test_BuyAndHolds": test_BuyAndHolds,
                              # Hyper-Hyperparams
                              "DV_colname": DV_colname,
                              "DV_shifts": [DV_shifts]*len(counts), "IV_lags": [IV_lags]*len(counts), 
                              "n_train_days": [n_train_days]*len(counts), "n_valid_days": [n_valid_days]*len(counts), "n_test_days": [n_test_days]*len(counts),
                              # Computation Costs
                              "execution_time": list_execution_cv}).set_index("CV")
    
    # Calculate the average feature importances of each variable
    if feature_importances:
        fi = pd.DataFrame(fi.T.describe().T["mean"].sort_values(ascending=False))
        fi = fi.rename({"mean": "average_value_of_feature_importances"}, axis=1)
        fi.to_csv(path+"/07_model_output/"+model_name+"_"+finished_time+"_"+"feature_importances.csv")
        display(fi.head(10))
    else:
        pass
    
    # Store the result table
      # Create the result file
    df_output.to_csv(path+"/07_model_output/"+model_name+"_"+finished_time+"_results_table.csv")
    
    # Make a folder to store the figures
    new_folder = path+"/08_reporting/"+model_name+"_"+finished_time
    os.makedirs(new_folder, exist_ok=False)
    
    # Visualize the results 
    viz_results(df, DV_colname, ModelPred_series, y_test_preprocessed, test_ModelScores, test_RandomScores, test_AlgoReturns, test_BuyAndHolds, test_Randoms, fi, feature_importances, new_folder, model_name, finished_time)
    
    # Save all figures individually
    save_figs(df, DV_colname, ModelPred_series, y_test_preprocessed, test_ModelScores, test_RandomScores, test_AlgoReturns, test_BuyAndHolds, test_Randoms, fi, feature_importances, new_folder, model_name, finished_time)
    
    return df_output
    

