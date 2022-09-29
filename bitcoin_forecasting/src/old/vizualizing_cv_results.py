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

path = "/content/drive/My Drive/Master_Research/workspace/data"

# Import the user-defined modules
import sys
sys.path.append(path+'/../src')
import preprocessing_ts, conducting_EDA
from preprocessing_ts import *
from conducting_EDA import *

def viz_results(df, DV_colname, ModelPred_series, y_test_preprocessed, cv_results, new_folder, model_name, feature_selection, DV_shifts, IV_lags, finished_time):
    
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
    # plt.savefig(new_folder+"/"+model_name+"_"+"FeatureSelection"+str(feature_selection)+"DVshifts"+str(DV_shifts)+"_"+"IVlags"+str(IV_lags)+"_"+finished_time+"_"+"all_figures.png", format="png", dpi=400)
    plt.savefig(new_folder+"/"+model_name+"_"+"FeatureSelection"+str(feature_selection)+"_DVshifts"+str(DV_shifts)+"_"+"IVlags"+str(IV_lags)+"_all"+".png", format="png", dpi=400)


def save_figs(df, DV_colname, ModelPred_series, y_test_preprocessed, cv_results, new_folder, model_name, feature_selection, DV_shifts, IV_lags, finished_time):
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
    plt.savefig(new_folder+"/"+model_name+"_"+"FeatureSelection"+str(feature_selection)+"_"+"DVshifts"+str(DV_shifts)+"_"+"IVlags"+str(IV_lags)+"_confusion_heatmap"+".png", format="png", dpi=400);
    
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
    plt.savefig(new_folder+"/"+model_name+"_"+"FeatureSelection"+str(feature_selection)+"_"+"DVshifts"+str(DV_shifts)+"_"+"IVlags"+str(IV_lags)+"_classification_report"+".png", format="png", dpi=400);
    
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
    plt.savefig(new_folder+"/"+model_name+"_"+"FeatureSelection"+str(feature_selection)+"_"+"DVshifts"+str(DV_shifts)+"_"+"IVlags"+str(IV_lags)+"_plot_accuracy"+".png", format="png", dpi=400);
    
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
    plt.savefig(new_folder+"/"+model_name+"_"+"FeatureSelection"+str(feature_selection)+"_"+"DVshifts"+str(DV_shifts)+"_"+"IVlags"+str(IV_lags)+"_dist_accuracy"+".png", format="png", dpi=400);
    
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
    plt.savefig(new_folder+"/"+model_name+"_"+"FeatureSelection"+str(feature_selection)+"_"+"DVshifts"+str(DV_shifts)+"_"+"IVlags"+str(IV_lags)+"_plot_returns"+".png", format="png", dpi=400);
    
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
    # plt.savefig(new_folder+"/"+model_name+"_"+"FeatureSelection"+str(feature_selection)+"_"+"DVshifts"+str(DV_shifts)+"_"+"IVlags"+str(IV_lags)+"_"+finished_time+"_"+"dist_returns"+".png", format="png", dpi=400);
    plt.savefig(new_folder+"/"+model_name+"_"+"FeatureSelection"+str(feature_selection)+"_"+"DVshifts"+str(DV_shifts)+"_"+"IVlags"+str(IV_lags)+"_dist_returns"+".png", format="png", dpi=400);
    
    # Make a folder to store the figures
#    new_folder = path+"/07_model_output/graphs/"+model_name+"_"+"FeatureSelection"+str(feature_selection)+"_"+"DVshifts"+str(DV_shifts)+"_"+"IVlags"+str(IV_lags)
#    os.makedirs(new_folder, exist_ok=True)
    
#    # Visualize the results 
#    viz_results(df, DV_colname, ModelPred_series, y_test_preprocessed, cv_results, new_folder, model_name, feature_selection, DV_shifts, IV_lags, finished_time)
    
#    # Save all figures individually
#    save_figs(df, DV_colname, ModelPred_series, y_test_preprocessed, cv_results, new_folder, model_name, feature_selection, DV_shifts, IV_lags, finished_time)