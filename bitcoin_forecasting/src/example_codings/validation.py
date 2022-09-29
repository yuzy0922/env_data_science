#ライブラリの宣言
import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error,r2_score


#docstringを追記
def time_series_split(y,train_start):
    
    """
    時系列と時間を与えると、学習データ2年分、検証用データ2か月分、テスト用データ2か月分に分ける

    Parameters:
    ----------
    y : データフレーム、データシリーズ

    train_start : 学習期間の最初の日付
    
    Returns
    -------
    train : 学習用データ
    valid : 検証用データ
    test : テスト用データ
    test_end : 最後の日付

    """
    
    #使う変数を宣言
    year2 = datetime.timedelta(days=365*2)
    month2 = datetime.timedelta(days=60)

    #学習用データを作る
    train_end = train_start+year2
    train = y[(train_start <= y.index) & (y.index < train_end)]

    #検証用データを作る
    valid_start = train_end
    valid_end = valid_start + month2
    valid = y[(valid_start <= y.index) & (y.index < valid_end)]

    #テスト用データを作る
    test_start = valid_end
    test_end = test_start + month2
    test = y[(test_start <= y.index) & (y.index < test_end)]
    
    return train,valid,test,test_end

def output_result(answer,predict,metric=None):
    """
    指定された文字列に合わせて評価指標を算出
    
    Parameters:
    -------
    answer : 正解
    predict : 予測
    metric : 評価指標（'MAE','MSE','MSLE','r2_score','RMSE','RMSLE'から選択）
    
    Return:
    -------
    算出結果
    
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

def rolling_window_valid(extracted_retail,modelling):
    
    """
    2年をtrain、2か月をvalid、2か月をtestとして与えられたデータを繰り返し予測していく

    Parameters:
    ----------
    extracted_retail : データフレーム、データシリーズ、2年4か月以上の時系列データ

    modelling : modelling=の形でモデリングするアルゴリズムを指定する
    
    Returns
    -------
    train_valid_results : Train+Validの結果を複数格納してあるもの
    test_results : testの評価結果を複数格納してあるもの
    predicts : テスト期間の予測をつなぎ合わせたもの

    """
    
    #変数の宣言
    y = pd.DataFrame(extracted_retail['Weekly_Sales'])
    train_start = y.index[0]
    test_end = train_start

    #繰り返しごとに出てくる結果の格納先
    train_valid_results = []
    test_results = []
    predicts = pd.DataFrame([])

    #検証の繰り返し
    while test_end < extracted_retail.index[-1]:

        #データの分割
        train,valid,test,test_end = time_series_split(y,train_start)
        
        #ハイパラ＋予測
        predict = modelling(train,valid,test)
        
        #trainとvalidを結合
        train_valid = pd.concat([train,valid])

        #結果の評価
        train_valid_result = output_result(train_valid[predict.index[0]:],predict[:train_valid.index[-1]],metric='RMSLE')
        test_result = output_result(test,predict[test.index[0]:],metric='RMSLE')

        #結果を格納します
        train_valid_results.append(train_valid_result)
        test_results.append(test_result)
        predicts = pd.concat([predicts,predict[test.index[0]:]])
        
        #次のために2か月足して回す
        train_start += datetime.timedelta(days=60)#最初の日にちに毎回60日足していく
    
    return train_valid_results,test_results,predicts

def rolling_window_valid_multi(extracted_retail,modelling):
    
    """
    2年をtrain、2か月をvalid、2か月をtestとして与えられたデータを繰り返し予測していく（多変量）

    Parameters:
    ----------
    extracted_retail : データフレーム、データシリーズ、2年4か月以上の多変量時系列データ

    modelling : modelling=の形でモデリングするアルゴリズムを指定する
    
    Returns
    -------
    train_valid_results : Train+Validの結果を複数格納してあるもの
    test_results : testの評価結果を複数格納してあるもの
    predicts : テスト期間の予測をつなぎ合わせたもの

    """
    
    #-------------rolling_window_validと異なる点-----------------
    train_start = extracted_retail.index[0]
    train,valid,test,test_end = time_series_split(extracted_retail,train_start)
    #---------------------------------------------------

    #何度も回すので。結果の格納先を定義
    train_valid_results = []
    test_results = []
    predicts = pd.DataFrame([])

    #ここから繰り返し
    while test_end < extracted_retail.index[-1]:

        train,valid,test,test_end = time_series_split(extracted_retail,train_start)
        predict = modelling(train,valid,test)
        train_valid = pd.concat([train,valid])

        #-------------rolling_window_validと異なる点-----------------
        #結果を算出
        train_valid_result = output_result(train_valid.loc[predict.index[0]:,'Weekly_Sales'],predict[:valid.index[-1]],metric='RMSLE')
        test_result = output_result(test['Weekly_Sales'],predict[test.index[0]:],metric='RMSLE')
        #---------------------------------------------------
        
        #結果の格納
        train_valid_results.append(train_valid_result)
        test_results.append(test_result)
        predicts = pd.concat([predicts,predict[test.index[0]:]])
        
        #次のために1か月足して回す
        train_start += datetime.timedelta(days=60)
        
    return train_valid_results,test_results,predicts