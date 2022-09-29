import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from lightgbm import *
from sklearn.ensemble import *
import tensorflow as tf
from tensorflow import keras
from boruta import BorutaPy
import optuna
from sklearn import metrics
import time

path = "..\data"

# Import the user-defined modules
import sys
sys.path.append(path+'/../src')
import preprocess_ts, conduct_eda, prepare_for_rwcv
from preprocess_ts import *
from conduct_eda import *
from prepare_for_rwcv import *


# Logistic
def Logistic(train, valid, test, DV_colname, DV_shifts, IV_lags, metric, feature_selection):
    """
    Create the model
    
    Params:
        train, valid, test: pd.DataFrame (Splitted data)
        DV_colname: str (The name best_of dependent variable used here)
        DV_shifts: The number of lags between DV and IVs

    Returns:
        X_train_preprocessed,y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed: pd.DataFrame (Preprocessed for modelling)
        optimized_model: the model whose hyperparams are optimized
        best_params: str (Dictionary of best_params)
    """

    # Preprocess for modelling
    full_preprocessed, X_train_preprocessed, y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed, X_train_valid, y_train_valid = \
        preprocess_for_modelling(train, valid, test, DV_colname, DV_shifts, IV_lags, feature_selection)

    # Optimize hyperparams of Logistic
    def objective(trial):
        solver = trial.suggest_categorical("solver", ['lbfgs', 'liblinear'])
        C = trial.suggest_loguniform("C", 1e-3, 1e0)
        model = LogisticRegression(solver=solver, C=C, random_state=201909)
        model.fit(X_train_preprocessed, y_train_preprocessed)
        pred_valid = np.round(model.predict(X_valid_preprocessed).flatten())
        return metric(y_valid_preprocessed, pred_valid)
      # Create a study object and optimize the objective function.
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=201909))
    study.optimize(objective, n_trials=100)

    # Choose the best params and store them to the model
    best_params = study.best_params
    optimized_model = LogisticRegression(**best_params, random_state=201909)

    # Fit the model to the train_valid
    optimized_model.fit(X_train_valid, y_train_valid)

    # Make predictions
    ModelPred_train_valid = np.round(optimized_model.predict(X_train_valid).flatten())
    ModelPred_test = np.round(optimized_model.predict(X_test_preprocessed).flatten())

    return optimized_model, best_params, full_preprocessed, X_train_valid, y_train_valid, X_test_preprocessed, y_test_preprocessed, ModelPred_train_valid, ModelPred_test

# NBC
def NBC(train, valid, test, DV_colname, DV_shifts, IV_lags, metric, feature_selection):
    """
    Create the model
    
    Params:
        train, valid, test: pd.DataFrame (Splitted data)
        DV_colname: str (The name best_of dependent variable used here)
        DV_shifts: The number of lags between DV and IVs

    Returns:
        X_train_preprocessed,y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed: pd.DataFrame (Preprocessed for modelling)
        optimized_model: the model whose hyperparams are optimized
        best_params: str (Dictionary of best_params)
    """

    # Preprocess for modelling
    full_preprocessed, X_train_preprocessed, y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed, X_train_valid, y_train_valid = \
        preprocess_for_modelling(train, valid, test, DV_colname, DV_shifts, IV_lags, feature_selection)

    # Optimize hyperparams of NBC
    def objective(trial):
        var_smoothing = trial.suggest_loguniform("var_smoothing", 1e-10, 1e1)
        model = GaussianNB(var_smoothing=var_smoothing)#, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        model.fit(X_train_preprocessed, y_train_preprocessed)
        pred_valid = np.round(model.predict(X_valid_preprocessed).flatten())
        return metric(y_valid_preprocessed, pred_valid)
      # Create a study object and optimize the objective function.
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=201909))
    study.optimize(objective, n_trials=100)

    # Choose the best params and store them to the model
    best_params = study.best_params
    optimized_model = GaussianNB(**best_params)

    # Fit the model to the train_valid
    optimized_model.fit(X_train_valid, y_train_valid)

    # Make predictions
    ModelPred_train_valid = np.round(optimized_model.predict(X_train_valid).flatten())
    ModelPred_test = np.round(optimized_model.predict(X_test_preprocessed).flatten())

    return optimized_model, best_params, full_preprocessed, X_train_valid, y_train_valid, X_test_preprocessed, y_test_preprocessed, ModelPred_train_valid, ModelPred_test
    
# SVMC
def SVMC(train, valid, test, DV_colname, DV_shifts, IV_lags, metric, feature_selection):
    """
    Create the model
    
    Params:
        train, valid, test: pd.DataFrame (Splitted data)
        DV_colname: str (The name best_of dependent variable used here)
        DV_shifts: The number of lags between DV and IVs

    Returns:
        X_train_preprocessed,y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed: pd.DataFrame (Preprocessed for modelling)
        optimized_model: the model whose hyperparams are optimized
        best_params: str (Dictionary of best_params)
    """

    # Preprocess for modelling
    full_preprocessed, X_train_preprocessed, y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed, X_train_valid, y_train_valid = \
        preprocess_for_modelling(train, valid, test, DV_colname, DV_shifts, IV_lags, feature_selection)

    # Optimize hyperparams of SVMC
    def objective(trial):
        C = trial.suggest_loguniform("C", 1e-6, 1e0)
        gamma = trial.suggest_loguniform("gamma", 1e-6, 1e0)
        kernel = trial.suggest_categorical("kernel", ['linear', 'poly', 'rbf', 'sigmoid'])
        degree = trial.suggest_int("degree", 1, 5)
        model = SVC(C=C, gamma=gamma, kernel=kernel, random_state=201909)
        model.fit(X_train_preprocessed, y_train_preprocessed)
        pred_valid = np.round(model.predict(X_valid_preprocessed).flatten())
        return metric(y_valid_preprocessed, pred_valid)
      # Create a study object and optimize the objective function.
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=201909))
    study.optimize(objective, n_trials=100)

    # Choose the best params and store them to the model
    best_params = study.best_params
    optimized_model = SVC(**best_params, random_state=201909)

    # Fit the model to the train_valid
    optimized_model.fit(X_train_valid, y_train_valid)

    # Make predictions
    ModelPred_train_valid = np.round(optimized_model.predict(X_train_valid).flatten())
    ModelPred_test = np.round(optimized_model.predict(X_test_preprocessed).flatten())

    return optimized_model, best_params, full_preprocessed, X_train_valid, y_train_valid, X_test_preprocessed, y_test_preprocessed, ModelPred_train_valid, ModelPred_test

# LGBMC
def LGBMC(train, valid, test, DV_colname, DV_shifts, IV_lags, metric, feature_selection):
    """
    Create the model
    
    Params:
        train, valid, test: pd.DataFrame (Splitted data)
        DV_colname: str (The name of dependent variable used here)
        DV_shifts: The number of differences between DV and IVs

    Returns:
        X_train_preprocessed,y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed: pd.DataFrame (Preprocessed for modelling)
        optimized_model: the model whose hyperparams are optimized
    """

    # Preprocess for modelling
    full_preprocessed, X_train_preprocessed, y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed, X_train_valid, y_train_valid = \
        preprocess_for_modelling(train, valid, test, DV_colname, DV_shifts, IV_lags, feature_selection)

    # Optimize hyperparams of LGBMC
    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 5, 10)
        n_estimators = trial.suggest_int("n_estimators", 1, 1000)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
        model = LGBMClassifier(max_depth=max_depth, n_estimators=n_estimators, 
                learning_rate=learning_rate, random_state=201909)
        model.fit(X_train_preprocessed, y_train_preprocessed)#, eval_set=(X_valid_preprocessed, y_valid_preprocessed), early_stopping_rounds=100)
        pred_valid = np.round(model.predict(X_valid_preprocessed).flatten())
        return metric(y_valid_preprocessed, pred_valid)
    # Create a study object and optimize the objective function.
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=201909))
    study.optimize(objective, n_trials=100)

    # Choose the best params and store them to the model
    best_params = study.best_params
    optimized_model = LGBMClassifier(**best_params, random_state=201909)

    # Fit the model to the train_valid
    optimized_model.fit(X_train_valid, y_train_valid)

    # Make predictions
    ModelPred_train_valid = np.round(optimized_model.predict(X_train_valid).flatten())
    ModelPred_test = np.round(optimized_model.predict(X_test_preprocessed).flatten())

    return optimized_model, best_params, full_preprocessed, X_train_valid, y_train_valid, X_test_preprocessed, y_test_preprocessed, ModelPred_train_valid, ModelPred_test
    
# RFC
def RFC(train, valid, test, DV_colname, DV_shifts, IV_lags, metric, feature_selection):
    """
    Create the model
    
    Params:
        train, valid, test: pd.DataFrame (Splitted data)
        DV_colname: str (The name best_of dependent variable used here)
        DV_shifts: The number of lags between DV and IVs

    Returns:
        X_train_preprocessed,y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed: pd.DataFrame (Preprocessed for modelling)
        optimized_model: the model whose hyperparams are optimized
        best_params: str (Dictionary of best_params)
    """

    # Preprocess for modelling
    full_preprocessed, X_train_preprocessed, y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed, X_train_valid, y_train_valid = \
        preprocess_for_modelling(train, valid, test, DV_colname, DV_shifts, IV_lags, feature_selection)

    # Optimize hyperparams of RFC
    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 3, 10)
        n_estimators = trial.suggest_int("n_estimators", 1, 1000)
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=201909)
        model.fit(X_train_preprocessed, y_train_preprocessed)
        pred_valid = np.round(model.predict(X_valid_preprocessed).flatten())
        return metric(y_valid_preprocessed, pred_valid)
      # Create a study object and optimize the objective function.
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=201909))
    study.optimize(objective, n_trials=100)

    # Choose the best params and store them to the model
    best_params = study.best_params
    optimized_model = RandomForestClassifier(**best_params, random_state=201909)

    # Fit the model to the train_valid
    optimized_model.fit(X_train_valid, y_train_valid)

    # Make predictions
    ModelPred_train_valid = np.round(optimized_model.predict(X_train_valid).flatten())
    ModelPred_test = np.round(optimized_model.predict(X_test_preprocessed).flatten())

    return optimized_model, best_params, full_preprocessed, X_train_valid, y_train_valid, X_test_preprocessed, y_test_preprocessed, ModelPred_train_valid, ModelPred_test
    
# FNN
def FNN(train, valid, test, DV_colname, DV_shifts, IV_lags, metric, feature_selection):
    """
    Create the model
    
    Params:
        train, valid, test: pd.DataFrame (Splitted data)
        DV_colname: str (The name of dependent variable used here)
        DV_shifts: The number of differences between DV and IVs

    Returns:
        X_train_preprocessed,y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed: pd.DataFrame (Preprocessed for modelling)
        optimized_model: the model whose hyperparams are optimized
    """

    # Preprocess for modelling
    full_preprocessed, X_train_preprocessed, y_train_preprocessed, X_valid_preprocessed, y_valid_preprocessed, X_test_preprocessed, y_test_preprocessed, X_train_valid, y_train_valid = \
        preprocess_for_modelling(train, valid, test, DV_colname, DV_shifts, IV_lags, feature_selection)

    # Optimize hyperparams
    def create_FNN(num_unit_hidden1, num_layer_hidden, num_unit_hidden, activation, dropout_rate):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(num_unit_hidden1, activation=activation, input_shape=(len(X_train_preprocessed.columns),))) # Input
        for i in range(num_layer_hidden):
            model.add(keras.layers.Dense(num_unit_hidden, activation=activation)) # Hidden
            model.add(keras.layers.Dropout(rate=dropout_rate)) # Dropout
        model.add(keras.layers.Dense(1, activation='sigmoid')) # Output
        return model
        
    def scheduler(epoch):
        if epoch < 10:
            return 1e-3
        else:
            return 1e-3 * tf.math.exp(0.1 * (10 - epoch))
    callback = keras.callbacks.LearningRateScheduler(scheduler)
    #callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    # Objective function
    def objective(trial):
        # Clear sessions
        keras.backend.clear_session()

        # Optimizing Hyperparams related to Network structure
            # the number of layers
        num_layer_hidden = trial.suggest_int("num_layer_hidden", 2, 4)
            # the numbers of input units, hidden units 
        num_unit_hidden1 = trial.suggest_int("num_unit_hidden1", 32, 128)
        num_unit_hidden = trial.suggest_int("num_unit_hidden", 8, 32)
            # activation function
        activation = trial.suggest_categorical("activation", [#"relu", 
                                                              #"sigmoid", 
                                                              "tanh",
                                                             ])
        
            # dropout_rate
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.4)
        # Optimizing Hyperparams related to Training Algorithm
            # optimizer
        optimizer = trial.suggest_categorical("optimizer", [#"sgd",
                                                            "adam",
                                                            #"rmsprop"
                                                            ])
            # batch size
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
            # epochs
        epochs = trial.suggest_int("epochs", 10, 30)

        # Create the model
        model = create_FNN(num_unit_hidden1, num_layer_hidden, num_unit_hidden, activation, dropout_rate)
        # Compile the model
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        # Learning
        history = model.fit(X_train_preprocessed.values, y_train_preprocessed.values, validation_data=(X_valid_preprocessed.values, y_valid_preprocessed.values), batch_size=batch_size, epochs=epochs, verbose=False, callbacks=[callback])
        # Explore the patterns of hyperparams which optimize the returned val_loss at the most
        pred_valid = np.round(model.predict(X_valid_preprocessed).flatten())
        return metric(y_valid_preprocessed, pred_valid)

    # Create a study object and optimize the objective function.
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=201909))
    study.optimize(objective, n_trials=100)

    # Store the best params
    best_params = study.best_params
    # Construct the model based on best_params
    optimized_model = keras.models.Sequential()
    # Create the model
    optimized_model.add(keras.layers.Dense(best_params["num_unit_hidden1"], activation=best_params["activation"], input_shape=(len(X_train_preprocessed.columns),))) # Input
    for i in range(best_params["num_layer_hidden"]):
        optimized_model.add(keras.layers.Dense(best_params["num_unit_hidden"], activation=best_params["activation"])) # Hidden
        optimized_model.add(keras.layers.Dropout(rate=best_params["dropout_rate"])) # Dropout
    optimized_model.add(keras.layers.Dense(1, activation='sigmoid')) # Output
    # Compile the model
    optimized_model.compile(loss="binary_crossentropy", optimizer=best_params["optimizer"], metrics=['accuracy'])
    # Learning
    history = optimized_model.fit(X_train_valid.values, y_train_valid.values, validation_data=(X_test_preprocessed.values, y_test_preprocessed.values), batch_size=best_params["batch_size"], epochs=best_params["epochs"], verbose=False, callbacks=[callback])

    # Make predictions
    ModelPred_train_valid = np.round(optimized_model.predict(X_train_valid).flatten())
    ModelPred_test = np.round(optimized_model.predict(X_test_preprocessed).flatten())
   
    # Summarize the model
    optimized_model.summary()
    # Plot loss per iteration
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.show()
    # Plot accuracy per iteration
    plt.plot(history.history["accuracy"], label="acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.show()

    return optimized_model, best_params, full_preprocessed, X_train_valid, y_train_valid, X_test_preprocessed, y_test_preprocessed, ModelPred_train_valid, ModelPred_test
    
