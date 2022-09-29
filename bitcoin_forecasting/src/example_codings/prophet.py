##ライブラリの宣言
import numpy as np
import pandas as pd
from fbprophet import Prophet
import warnings
warnings.simplefilter('ignore')

#前処理を行う関数
def prophet_preprocess(df):
    
    """
    prophetに入れるようにデータフレームの前処理を行います
    
    Returns:
    ------
    ret : 前処理したデータフレーム
    """
    
    ret = df.copy()#データフレームをコピーする
    
    #対数変換
    ret['Weekly_Sales'] = np.log1p(ret['Weekly_Sales'])
    
    #列名の変更
    ret['ds'] = ret.index
    ret = ret.rename(columns={'Weekly_Sales':'y'})
    
    return ret

##モデリングを行う関数
def prophet_modelling(train,valid,test):
    
    """
    prophetによる多変量時系列の予測を行います。
    
    Parameters:
    ------
    train,valid,test : rolling_window_valid関数で得られるtrain,valid,testデータ
    
    Return:
    ------
    predict :できる限りの区間予測したものの予測（学習期間もテスト期間も含む）
    
    """
    
    #前処理
    pre_train = prophet_preprocess(train)
    pre_valid = prophet_preprocess(valid)
    pre_test = prophet_preprocess(test)
    
    #前処理　holidaysの設定
    pre_full = pd.concat([pre_train,pre_valid,pre_test])
    holidays = pre_full[['ds','IsHoliday']]
    holidays = holidays.rename(columns={'IsHoliday':'holiday'})
    holidays['holiday'] = holidays['holiday'].astype('str')

    #モデルの設定
    model = Prophet(yearly_seasonality = True, 
                    weekly_seasonality = False,
                    daily_seasonality = False,
                   holidays=holidays) #holidaysを設定

    model.add_country_holidays(country_name='US')#アメリカの休日も挿入

    model.add_regressor('Temperature')#気温のみ外部変数として追加

    #使うデータを定義
    pre_train_valid = pd.concat([pre_train,pre_valid])
    model.fit(pre_train_valid)

    # 予測期間の設定
    future_term = pre_full[['ds','Temperature']].reset_index(drop=True)

    # 予測を出力する
    forecast = model.predict(future_term)

    predict = pd.DataFrame(data=forecast['yhat'].values,index=forecast['ds'].values)
    predict = np.expm1(predict)
    
    return predict