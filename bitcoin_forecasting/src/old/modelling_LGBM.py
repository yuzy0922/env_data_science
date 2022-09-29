import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import constructing_experiment
from constructing_experiment import split_ts, check_output_accuracy, rolling_window_validation

def make_lgb_dataset(df, DV_colname):

    """
    Adjust the dataframe to LGBM model

    Params:
    -----
    df: pd.DataFrame
    DV_colname: str (The column name of the dependent variable focused in the experiment)

    Return:
    -----
    X: pd.DataFrame (only includes Independent variables)
    Dataset: the dataset for LGBM

    """

    X = df.drop([DV_colname], axis=1)
    y = df[DV_colname]

    return X, lgb.Dataset(X, y)
    

def build_lgb_model(train, valid, test, DV_colname, DV_shifts):
    """
    Make a prediction for LGBM multivariate.

    Params:
        train, valid, test: pd.DataFrame (The data divided by the function "rolling_window_validation")
        DV_colname: str (The column name of the dependent variable focused in the experiment)
        DV_shifts: int (How many lags of DV and IVs we assume.)

    Return:
        predicted: pd.Series (All predicted results including train, valid, test periods)

    """

    # Merge the data
    full = pd.concat([train, valid, test])

    # Shift the IVs by n because they are not accessible prior to 2 or 3 days.
    full[full.columns[~full.columns.isin([DV_colname])]] = full.drop(DV_colname,axis=1).shift(DV_shifts)
    full = full[DV_shifts:]

    # Split the data again because preprocessing for LGBM was finished!
    train_preprocessed = full[:train.index[-1]]
    valid_preprocessed = full[valid.index[0]: valid.index[-1]]
    test_preprocessed = full[test.index[0]: test.index[-1]]

    # Create LGBM dataset
    X_train_preprocessed, lgb_train_preprocessed = make_lgb_dataset(train_preprocessed, DV_colname)
    X_valid_preprocessed, lgb_valid_preprocessed = make_lgb_dataset(valid_preprocessed, DV_colname)
    X_test_preprocessed, lgb_test_preprocessed = make_lgb_dataset(test_preprocessed, DV_colname)

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "seed": 0
    }

    bst = lgb.train(
        lgb_params,
        train_set = lgb_train_preprocessed,
        valid_sets = lgb_valid_preprocessed,
        num_boost_round=1000,
        early_stopping_rounds=10,
        verbose_eval=True
    )

    # Merge train and valid
    train_valid_preprocessed = pd.concat([train_preprocessed, valid_preprocessed])
    _, lgb_train_valid_preprocessed = make_lgb_dataset(train_valid_preprocessed, DV_colname)

    # Execute learning again
    best_bst = lgb.train(
        params=lgb_params, 
        train_set=lgb_train_valid_preprocessed, 
        num_boost_round=bst.best_iteration
    )

    # Merge the all of independent variables in train, valid and test periods
    X_full = pd.concat([X_train_preprocessed, X_valid_preprocessed, X_test_preprocessed])

    # Make a prediction
    predicted = best_bst.predict(X_full)

    # Change the index of "predicted"
    predicted = pd.Series(predicted)
    predicted.index = list(train.index[DV_shifts:]) + list(valid.index) + list(test.index)

    # Plot the time series of answer, predicted in valid and predicted in test
    plt.figure(figsize=(8,4))
    plt.plot(full.loc[:test.index[-1], DV_colname])
    plt.plot(predicted[:valid.index[-1]])
    plt.plot(predicted[test.index[0]:])
    plt.legend(["answer", "predicted", "predicted_test"])
    plt.show()

    # Interpretation of accuracy metrics in valid and test periods
    train_valid_result = check_output_accuracy(pd.concat([train[DV_shifts:], valid])[DV_colname], predicted[:valid.index[-1]], metric="RMSE")
    test_result = check_output_accuracy(test[DV_colname], predicted[test.index[0]:], metric="RMSE")
    print("Train:", train_valid_result)
    print("Test:", test_result)

    return predicted