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
        full_preprocessed, X_train_preprocessed, y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed, X_train_valid, y_train_valid: splitted and preprocessed data
    
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

    # StandardScaling inputs
      # X_train --> X_valid
#    if full_preprocessed.max().max()!=1.0:
    StdSc_train = StandardScaler()
    X_train_preprocessed = pd.DataFrame(StdSc_train.fit_transform(X_train_preprocessed),columns = X_train_preprocessed.columns)
    X_valid_preprocessed = pd.DataFrame(StdSc_train.transform(X_valid_preprocessed),columns = X_valid_preprocessed.columns)
      # X_train_valid --> X_test
    StdSc_train_valid = StandardScaler()
    X_train_valid = pd.DataFrame(StdSc_train_valid.fit_transform(X_train_valid),columns = X_train_valid.columns)
    X_test_preprocessed = pd.DataFrame(StdSc_train_valid.transform(X_test_preprocessed),columns = X_test_preprocessed.columns)
    
    return full_preprocessed, X_train_preprocessed, y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed, X_train_valid, y_train_valid
    
def viz_results(df, DV_colname, ModelPred_series, y_test_preprocessed, cv_results, new_folder, model_name, finished_time):
    
    # Calculate the average forecasting scores
    print("Average of forecasting-model scores on test", cv_results["test_model_accuracy"].mean()) 
    print("Average of random-forecasting scores on test", cv_results["test_random_accuracy"].mean())   
    print("Average of naive-forecasting scores on test", cv_results["test_naive_accuracy"].mean())     

    # Calculate the average of total returns on each CV
    print("Average of total Algo return on test", cv_results["test_AlgoReturns"].mean())       
    print("Average of total Random return on test", cv_results["test_RandomReturns"].mean())    
    print("Average of total Naive forecasting return on test", cv_results["test_NaiveReturns"].mean())     
    print("Average of total Buy-And-Hold return on test", cv_results["test_BuyAndHolds"].mean())
   
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
    ax3.plot(cv_results["test_model_accuracy"], label="ModelPred")
    ax3.plot(cv_results["test_random_accuracy"], label="RandomPred")
    ax3.plot(cv_results["test_naive_accuracy"], label="NaivePred")
    ax3.set_title("Forecasting Accuracies on each CV")
    ax3.set_xlabel("Times of Cross Validation")
    ax3.set_xticks([item for item in range(0, len(cv_results)+1)])
    ax3.set_ylabel("Forecasting Accuracies")
    ax3.grid(True)
    ax3.legend()
    
    # Density Plot
    sns.distplot(cv_results["test_model_accuracy"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, ax=ax4)
    sns.distplot(cv_results["test_random_accuracy"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, ax=ax4)
    sns.distplot(cv_results["test_naive_accuracy"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, ax=ax4)
    ax4.set_title("Density Plot of Forecasting Accuracies on each CV")
    ax4.set_xlabel("Forecasting Accuracy")
    ax4.set_ylabel("Density Function")
    ax4.grid(True)
    ax4.legend(labels=["ModelPred", "RandomPred", "NaivePred"])

    # Visualize the total returns
    ax5.plot(cv_results["test_AlgoReturns"], label="ModelPred")
    ax5.plot(cv_results["test_RandomReturns"], label="RandomPred")
    ax5.plot(cv_results["test_NaiveReturns"], label="NaivePred")
    ax5.plot(cv_results["test_BuyAndHolds"], label="Buy-and-Hold")    
    ax5.set_title("Total Returns on each CV")
    ax5.set_xlabel("Times of Cross Validation")
    ax5.set_xticks([item for item in range(0, len(cv_results)+1)])
    ax5.set_ylabel("Total Returns")
    ax5.grid(True)
    ax5.legend()
    
    # Density Plot
    sns.distplot(cv_results["test_AlgoReturns"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, ax=ax6)
    sns.distplot(cv_results["test_RandomReturns"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, ax=ax6)
    sns.distplot(cv_results["test_NaiveReturns"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, ax=ax6)
    sns.distplot(cv_results["test_BuyAndHolds"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, ax=ax6)
    ax6.set_title("Density Plot of Total Returns on each CV")
    ax6.set_xlabel("Total Returns")
    ax6.set_ylabel("Density Function")
    ax6.grid(True)
    ax6.legend(labels=["ModelPred", "RandomPred", "NaivePred", "Buy-and-Hold"])
    
    fig.tight_layout()
    fig.show()
    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"all_figures.png", format="png", dpi=400)


def save_figs(df, DV_colname, ModelPred_series, y_test_preprocessed, cv_results, new_folder, model_name, finished_time):
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
    plt.tight_layout()
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
    plt.tight_layout()
    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"classification_report"+".png", format="png", dpi=400);
    
    # ax3
    plt.figure(figsize=(12,8))
    plt.plot(cv_results["test_model_accuracy"], label="ModelPred")
    plt.plot(cv_results["train_valid_random_accuracy"], label="RandomPred")
    plt.plot(cv_results["test_random_accuracy"], label="NaivePred")
    plt.title("Forecasting Accuracies on each CV")
    plt.xlabel("Times of Cross Validation")
    plt.xticks([item for item in range(0, len(cv_results)+1)])
    plt.ylabel("Forecasting Accuracies")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"plot_accuracy"+".png", format="png", dpi=400);
    
    # ax4
#      # Histgram
#    plt.figure(figsize=(12,8))
#    plt.hist(test_ModelAccuracies, bins=6, label="ModelPred", alpha=0.3)
#    plt.hist(test_RandomAccuracies, bins=6, label="RandomPred", alpha=0.3)
#    plt.title("Distribution of Forecasting Accuracies on each CV")
#    plt.xlabel("Forecasting Accuracy")
#    plt.ylabel("Frequency")
#    plt.grid(True)
#    plt.legend()
#    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"dist_accuracy"+".png", format="png", dpi=400);
      # Density Plot
    plt.figure(figsize=(12,8))
    sns.distplot(cv_results["test_model_accuracy"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})
    sns.distplot(cv_results["test_random_accuracy"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})
    sns.distplot(cv_results["test_naive_accuracy"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})
    plt.title("Density Plot of Forecasting Accuracies on each CV")
    plt.xlabel("Forecasting Accuracy")
    plt.ylabel("Density Function")
    plt.grid(True)
    plt.legend(labels=["ModelPred", "RandomPred", "NaivePred"])
    plt.tight_layout()
    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"dist_accuracy"+".png", format="png", dpi=400);
    
    # ax5
    plt.figure(figsize=(12,8))
    plt.plot(cv_results["test_AlgoReturns"], label="ModelPred")
    plt.plot(cv_results["test_RandomReturns"], label="RandomPred")
    plt.plot(cv_results["test_NaiveReturns"], label="NaivePred")
    plt.plot(cv_results["test_BuyAndHolds"], label="Buy-and-Hold")
    plt.title("Total Returns on each CV")
    plt.xlabel("Times of Cross Validation")
    plt.xticks([item for item in range(0, len(cv_results)+1)])
    plt.ylabel("Total Returns")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"plot_returns"+".png", format="png", dpi=400);
    
    # ax6
      # Density Plot
    plt.figure(figsize=(12,8))
    sns.distplot(cv_results["test_AlgoReturns"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})
    sns.distplot(cv_results["test_RandomReturns"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})
    sns.distplot(cv_results["test_NaiveReturns"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})
    sns.distplot(cv_results["test_BuyAndHolds"], hist=False, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})
    plt.title("Density Plot of Total Returns on each CV")
    plt.xlabel("Total Returns")
    plt.ylabel("Density Function")
    plt.grid(True)
    plt.legend(labels=["ModelPred", "RandomPred", "NaivePred", "Buy-and-Hold"])
    plt.tight_layout()
    plt.savefig(new_folder+"/"+model_name+"_"+finished_time+"_"+"dist_returns"+".png", format="png", dpi=400);


# Execute rwcv_c
def rwcv_c(df, model_name, modelling, DV_colname="btc_LogReturn_binary", DV_shifts=1, IV_lags=2, n_train_days=540, n_valid_days=30, n_test_days=30, metric=f1_score, feature_importances=False):
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
        output_results: pd.DataFrame (the table of experimental results)

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
    
    # Pred Series
    ModelPred_series = pd.DataFrame([])
    
    # Simulated Returns
    train_valid_AlgoReturns = []
    test_AlgoReturns = []
    train_valid_RandomReturns = []
    test_RandomReturns = []
    train_valid_NaiveReturns = []
    test_NaiveReturns = []
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
        optimized_model, best_params, full_preprocessed, X_train_valid, y_train_valid, X_test_preprocessed, y_test_preprocessed, ModelPred_train_valid, ModelPred_test = \
            modelling(train, valid, test, DV_colname, DV_shifts, IV_lags, metric)
        
        # Concat train_valid and test pred series to create one series
        ModelPred_all = pd.concat([pd.Series(ModelPred_train_valid), pd.Series(ModelPred_test)])
        ModelPred_all.index = full_preprocessed.index
        ModelPred_series = pd.concat([ModelPred_series, ModelPred_all[test.index[0]:]])
    
        # Random prediction
        RandomPred_train_valid = np.round(np.random.rand(len(y_train_valid)))
        RandomPred_test = np.round(np.random.rand(len(y_test_preprocessed)))
        
        # Naive forecasting
        NaivePred_train_valid = (full_preprocessed[y_train_valid.index[0]: y_train_valid.index[-1]]["btc_LogReturn"] > 0).astype(int)
        NaivePred_test = (full_preprocessed[y_test_preprocessed.index[0]: y_test_preprocessed.index[-1]]["btc_LogReturn"] > 0).astype(int)

        
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
          # Take positions by the naive forecasting
        full_preprocessed["NaivePosition"] = 0
        full_preprocessed.loc[y_train_valid.index[0]: y_train_valid.index[-1], "NaivePosition"] = NaivePred_train_valid
        full_preprocessed.loc[y_test_preprocessed.index[0]:y_test_preprocessed.index[-1], "NaivePosition"] = NaivePred_test
        full_preprocessed["NaiveReturn"] = full_preprocessed["NaivePosition"] * df[full_preprocessed.index[0]: full_preprocessed.index[-1]]["btc_LogReturn"]
         
          # Calculate the sum of Algo-Returns
        train_valid_AlgoReturns.append(full_preprocessed[y_train_valid.index[0]:y_train_valid.index[-1]]["AlgoReturn"].sum())
        test_AlgoReturns.append(full_preprocessed[y_test_preprocessed.index[0]: y_test_preprocessed.index[-1]]["AlgoReturn"].sum())
          # Calculate the sum of random returns
        train_valid_RandomReturns.append(full_preprocessed[y_train_valid.index[0]:y_train_valid.index[-1]]["RandomReturn"].sum())
        test_RandomReturns.append(full_preprocessed[y_test_preprocessed.index[0]: y_test_preprocessed.index[-1]]["RandomReturn"].sum())
          # Calculate the sum of Naive forecasting returns
        train_valid_NaiveReturns.append(full_preprocessed[y_train_valid.index[0]: y_train_valid.index[-1]]["NaiveReturn"].sum())
        test_NaiveReturns.append(full_preprocessed[y_test_preprocessed.index[0]: y_test_preprocessed.index[-1]]["NaiveReturn"].sum())
          # Calculate the sum of Buy-and-Hold returns
        train_valid_BuyAndHolds.append(full_preprocessed[y_train_valid.index[0]: y_train_valid.index[-1]]["btc_LogReturn"].sum())
        test_BuyAndHolds.append(full_preprocessed[y_test_preprocessed.index[0]: y_test_preprocessed.index[-1]]["btc_LogReturn"].sum())
        
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
    cv_results = pd.DataFrame({"CV": counts, "test_start": test_starts, "test_end": test_ends, "model": [model_name]*len(counts), "best_params": list_best_params, "vars_set": [X_train_valid.columns]*len(counts),
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
                              # Profitability
                              "train_valid_AlgoReturns": train_valid_AlgoReturns, "test_AlgoReturns": test_AlgoReturns,
                              "train_valid_RandomReturns": train_valid_RandomReturns, "test_RandomReturns": test_RandomReturns,
                              "train_valid_NaiveReturns": train_valid_NaiveReturns, "test_NaiveReturns": test_NaiveReturns,                              
                              "train_valid_BuyAndHolds": train_valid_BuyAndHolds, "test_BuyAndHolds": test_BuyAndHolds,
                              # Hyper-Hyperparams
                              "DV_colname": [DV_colname]*len(counts),
                              "DV_shifts": [DV_shifts]*len(counts), "IV_lags": [IV_lags]*len(counts), 
                              "n_train_days": [n_train_days]*len(counts), "n_valid_days": [n_valid_days]*len(counts), "n_test_days": [n_test_days]*len(counts),
                              # Computation Costs
                              "execution_time": list_execution_cv}).set_index("CV")
    average_results = pd.DataFrame({"CV": ["Average"], "test_start": [test_starts[0]], "test_end": [test_ends[-1]], "model": [model_name], "best_params": [np.nan], "vars_set": [X_train_valid.columns], 
                              # Forecasting Scores
                              "train_valid_model_f1": [np.mean(train_valid_ModelF1s)], 
                              "test_model_f1": [np.mean(test_ModelF1s)],                               
                              "train_valid_random_f1": [np.mean(train_valid_RandomF1s)], 
                              "test_random_f1": [np.mean(test_RandomF1s)],
                              "train_valid_naive_f1": [np.mean(train_valid_NaiveF1s)], 
                              "test_naive_f1": [np.mean(test_NaiveF1s)],
                              "train_valid_model_accuracy": [np.mean(train_valid_ModelAccuracies)], 
                              "test_model_accuracy": [np.mean(test_ModelAccuracies)],                               
                              "train_valid_random_accuracy": [np.mean(train_valid_RandomAccuracies)], 
                              "test_random_accuracy": [np.mean(test_RandomAccuracies)],
                              "train_valid_naive_accuracy": [np.mean(train_valid_NaiveAccuracies)], 
                              "test_naive_accuracy": [np.mean(test_NaiveAccuracies)],
                              
                              # Profitability
                              "train_valid_AlgoReturns": [np.mean(train_valid_AlgoReturns)], "test_AlgoReturns": [np.mean(test_AlgoReturns)],
                              "train_valid_RandomReturns": [np.mean(train_valid_RandomReturns)], "test_RandomReturns": [np.mean(test_RandomReturns)],
                              "train_valid_NaiveReturns": [np.mean(train_valid_NaiveReturns)], "test_NaiveReturns": [np.mean(test_NaiveReturns)],
                              "train_valid_BuyAndHolds": [np.mean(train_valid_BuyAndHolds)], "test_BuyAndHolds": [np.mean(test_BuyAndHolds)],
                              # Hyper-Hyperparams
                              "DV_colname": [DV_colname],
                              "DV_shifts": [DV_shifts], "IV_lags": [IV_lags], 
                              "n_train_days": [n_train_days], "n_valid_days": [n_valid_days], "n_test_days": [n_test_days],
                              # Computation Costs
                              "execution_time": [np.mean(list_execution_cv)]}).set_index("CV")
    output_results = pd.concat([cv_results, average_results], axis=0)
    
    # Calculate the average feature importances of each variable
    if feature_importances:
        fi = pd.DataFrame(fi.T.describe().T["mean"].sort_values(ascending=False))
        fi = fi.rename({"mean": "average_value_of_feature_importances"}, axis=1)
        fi.to_csv(path+"/07_model_output/feature_importances/"+model_name+"_"+finished_time+"_"+"feature_importances.csv")
        display(fi.head(10))
    else:
        pass
    
    # Store the result table
      # Create the result file
    output_results.to_csv(path+"/07_model_output/result_table/"+model_name+"_"+finished_time+"_results_table.csv")
    
    # Make a folder to store the figures
    new_folder = path+"/08_reporting/"+model_name+"_"+finished_time
    os.makedirs(new_folder, exist_ok=False)
    
    # Visualize the results 
    viz_results(df, DV_colname, ModelPred_series, y_test_preprocessed, cv_results, new_folder, model_name, finished_time)
    
    # Save all figures individually
    save_figs(df, DV_colname, ModelPred_series, y_test_preprocessed, cv_results, new_folder, model_name, finished_time)
    
    return output_results
