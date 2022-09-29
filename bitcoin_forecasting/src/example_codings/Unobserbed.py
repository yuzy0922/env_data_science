#使うライブラリを宣言
from statsmodels.tsa.api import UnobservedComponents
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd

def UnobservedComponents_modelling(train,valid,test):
    
    """
    状態空間モデルによる単変量の予測を行います。当てはまりのいいformulaを毎回設定します
    
    Parameters:
    ------
    train,valid,test : rolling_window_valid関数で得られるtrain,valid,testデータ
    
    Return:
    ------
    predict :できる限りの区間予測したものの予測（学習期間もテスト期間も含む）
    
    """
    
    #対数変換
    log_train,log_valid,log_test = np.log(train),np.log(valid),np.log(test)
    
    formulas = ['irregular','fixed intercept','deterministic constant','local level','random walk','fixed slope',
                'deterministic trend','local linear deterministic trend','random walk with drift','local linear trend','smooth trend',
                'random trend']
    
    #結果格納用
    results = pd.DataFrame([],columns=['formula_name','aic'])

    #モデルごとに試して結果を格納する
    for formula in formulas:

        #学習
        model = UnobservedComponents(endog = log_train,
                                     level = formula).fit()
        #予測
        predict = model.predict(start = 0,
                                end = len(log_train)+len(log_valid)-1,
                                dynamic=False)
        #predictのインデックスを変える
        predict.index = list(train.index) + list(valid.index)
        predict = predict[1:]
        predict = np.expm1(predict)

        #結果を格納
        aic = model.aic
        result = np.array([formula,aic])
        result = pd.DataFrame(result.reshape(1,-1),columns=results.columns)
        results = results.append(result)

    #ベストなパラメータを決める
    best_formula = results[results['aic'] == results['aic'].min()]['formula_name'].values[0]
    
    #log_trainとlog_validを結合
    log_train_valid = pd.concat([log_train,log_valid])

    #結合データに対して、best_paramで学習
    final_model = UnobservedComponents(endog = log_train_valid,
                                       level = best_formula).fit()

    predict = final_model.predict(start = 0,
                                  end = len(log_train_valid)+len(log_test)-1,
                                  dynamic=False)

    #predictのインデックスを変える
    predict.index = list(log_train.index) + list(log_valid.index) + list(log_test.index)
    predict = np.expm1(predict)
    predict = predict[1:]

    return predict