#ライブラリのインポート
import warnings
warnings.simplefilter('ignore')
from statsmodels.tsa.api import SARIMAX
from tqdm import tqdm_notebook as tqdm
import itertools
import numpy as np
import pandas as pd

##モデリングを行う関数
def sarimax_modelling(train,valid,test):
    
    """
    SARIMAXモデルによる多変量時系列の予測を行います。
    
    Parameters:
    ------
    train,valid,test : rolling_window_valid関数で得られるtrain,valid,testデータ
    
    Return:
    ------
    predict :できる限りの区間予測したものの予測（学習期間もテスト期間も含む）
    
    """

    #MarkDownを除く
    col = ['Weekly_Sales', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI','Unemployment']
    #train,valid,test = train.fillna(0),valid.fillna(0),test.fillna(0)
    train,valid,test = train[col],valid[col],test[col]
    
    #対数変換
    train.loc[:,'log_Weekly_Sales'] = np.log1p(train['Weekly_Sales'])
    valid.loc[:,'log_Weekly_Sales'] = np.log1p(valid['Weekly_Sales'])
    test.loc[:,'log_Weekly_Sales'] = np.log1p(test['Weekly_Sales'])
    
    #パラメータの選択肢を与える
    p,d,q = range(3),range(3),range(3)
    sp,sd,sq,s = range(2),[0],range(2),[7]

    #パラメータの組み合わせを作る
    order_list = list(itertools.product(p,d,q))
    seasonal_order_list = list(itertools.product(sp,sd,sq,s))
    all_option = list(itertools.product(order_list,seasonal_order_list))
    
    
    #ハイパラチューニング
    results = pd.DataFrame([],columns=['order','seasonal_order','aic'])
    for option in tqdm(all_option):
        model = SARIMAX(endog=train['log_Weekly_Sales'],
                        exog=train.drop(['log_Weekly_Sales','Weekly_Sales'],axis=1),
                        order=option[0],
                        seasonal_order=option[1],
                        enforce_invertibility=False,
                        enforce_stationarity=False).fit()
        predict = model.predict(start = 0,
                            end = len(train)+len(valid)-1,
                            exog = valid.drop(['log_Weekly_Sales','Weekly_Sales'],axis=1),
                            dynamic=False)
        aic = model.aic
        result = np.array([option[0],option[1],aic])
        result = pd.DataFrame(result.reshape(1,-1),columns=results.columns)
        results = results.append(result)
    
    final_order = results[results['aic'] == results['aic'].min()]['order'].values[0]
    seasonal_order = results[results['aic'] == results['aic'].min()]['seasonal_order'].values[0]
    
    #もう一度学習
    train_valid = pd.concat([train,valid])
    model = SARIMAX(endog=train_valid['log_Weekly_Sales'],
                    exog=train_valid.drop(['log_Weekly_Sales','Weekly_Sales'],axis=1),
                    order=final_order,
                    seasonal_order=seasonal_order,
                    enforce_invertibility=False,
                    enforce_stationarity=False).fit()
    predict = model.predict(start = 0,
                            end = len(train_valid)+len(test)-1,
                            exog = test.drop(['log_Weekly_Sales','Weekly_Sales'],axis=1),
                            dynamic=False)
    
    #predictを整える
    predict.index = list(train.index) + list(valid.index) + list(test.index)
    predict = pd.DataFrame(predict)
    predict = np.expm1(predict)
    predict = predict[1:]
    
    return predict