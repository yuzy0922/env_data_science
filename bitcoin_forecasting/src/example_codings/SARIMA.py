#ライブラリインポート
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.simplefilter('ignore')
import itertools
from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd

def sarima_modelling(train,valid,test):
    """
    SARIMAモデルによる単変量の予測を行います。
    
    Parameters:
    ------
    train,valid,test : rolling_window_valid関数で得られるtrain,valid,testデータ
    
    Return:
    ------
    predict :できる限りの区間予測したものの予測（学習期間もテスト期間も含む）
    
    """
    
    
    #対数変換
    log_train,log_valid,log_test = np.log1p(train),np.log1p(valid),np.log1p(test)

    #パラメータの選択肢を与える
    p,d,q = range(3),range(3),range(3)
    sp,sd,sq,s = range(2),[0],range(2),[7]

    #パラメータの組み合わせを作る
    order_list = list(itertools.product(p,d,q))
    seasonal_order_list = list(itertools.product(sp,sd,sq,s))

    all_option = list(itertools.product(order_list,seasonal_order_list))
    
    #結果保存用データフレーム
    results = pd.DataFrame([],columns=['order','seasonal_order','aic'])

    #繰り返し
    for option in tqdm(all_option):
        #学習
        model = SARIMAX(endog=log_train,
                        order=option[0],
                        seasonal_order=option[1],
                       enforce_invertibility=False,
                       enforce_stationarity=False).fit()
        predict = model.predict(start = 0,
                                end = len(log_train)+len(log_valid)-1,
                                dynamic=False)
        #predictのindexを変える(前回と同様)
        predict.index = list(log_train.index) + list(log_valid.index)
        predict = predict[1:]
        predict = np.expm1(predict)
        #結果算出
        aic = model.aic
        #結果格納
        result = np.array([option[0],option[1],aic])
        result = pd.DataFrame(result.reshape(1,-1),columns=results.columns)
        results = results.append(result)

    #ハイパラ確定
    min_aic = results[results['aic'] == results['aic'].min()]
    best_order = min_aic['order'].values[0]
    best_seasonal_order = min_aic['seasonal_order'].values[0]
        
    #もう一度学習
    #log_trainとlog_validを結合
    log_train_valid = pd.concat([log_train,log_valid])

    #結合データに対して、best_paramで学習
    model = SARIMAX(endog=log_train_valid,
                    order=best_order,
                    seasonal_order=best_seasonal_order,
                    enforce_invertibility=False,
                    enforce_stationarity=False).fit()
    predict = model.predict(start = 0,
                            end = len(log_train_valid)+len(log_test)-1,
                            dynamic=False)
    
    #predictのインデックスを変える
    predict.index = list(log_train.index) + list(log_valid.index) + list(log_test.index)
    predict = predict[1:]
    predict = np.expm1(predict)
    
    return predict