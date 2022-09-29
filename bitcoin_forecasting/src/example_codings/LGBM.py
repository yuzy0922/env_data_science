#ライブラリの宣言
import numpy as np
import pandas as pd
import lightgbm as lgb
from validation import output_result

def lgb_preprocessing(train,valid,test):
    """
    LightGBMモデルの単変量予測のための特徴量作成を行う関数です
    
    Parameters:
    ------
    train,valid,test : rolling_window_validで分割されたデータ
    
    Returns:
    ------
    pp_train,pp_valid,pp_test : 前処理された各データ
    
    """
    
    #対数変換
    log_train,log_valid,log_test = np.log1p(train),np.log1p(valid),np.log1p(test)

    #データの結合
    log_full = pd.concat([log_train,log_valid,log_test])
    log_full = log_full.rename({'Weekly_Sales':'log_Weekly_Sales'},axis=1)

    #ラグを取る
    n = 5
    for i in range(1,n):
        log_full['log_lag'+str(i)] = log_full['log_Weekly_Sales'].shift(i)
        
    #移動平均と標準偏差を算出
    rolling_n = [4,8,10,20]
    for i in rolling_n:
        log_full['log_Weekly_Sales_rolling_mean_' + str(i)] = log_full['log_Weekly_Sales'].rolling(i).mean().shift()
        log_full['log_Weekly_Sales_rolling_std_' + str(i)] = log_full['log_Weekly_Sales'].rolling(i).std().shift()

    #対数収益率
    log_full['log_shuekiritu'] = log_full['log_Weekly_Sales'].diff().shift()
    
    
    #再度前処理したtrain,valid,testに分ける
    pp_train = log_full[train.index[0]:train.index[-1]]
    pp_valid = log_full[valid.index[0]:valid.index[-1]]
    pp_test = log_full[test.index[0]:test.index[-1]]
    
    return pp_train,pp_valid,pp_test

def make_lgb_dataset(df):
    
    """
    データフレームをLightGBM用のデータセットに変換します。
    
    Return:
    ------
    X : 説明変数のみのデータフレーム
    データセット : LightGBM用のデータセット
    
    """
    
    X = df.drop(['log_Weekly_Sales'],axis=1)
    y = df['log_Weekly_Sales']
    
    return X,lgb.Dataset(X,y)

def lgb_modelling(train,valid,test):
    
    """
    LightGBMによる単変量の予測を行います。num_boost_roundを自動的にチューニングします
    
    Parameters:
    ------
    train,valid,test : rolling_window_valid関数で得られるtrain,valid,testデータ
    
    Return:
    ------
    predict :できる限りの区間予測したものの予測（学習期間もテスト期間も含む）
    
    """
    
    #前処理
    pp_train,pp_valid,pp_test = lgb_preprocessing(train,valid,test)

    #データセットを作る
    X_pp_train, lgb_pp_train = make_lgb_dataset(pp_train)
    X_pp_valid, lgb_pp_valid = make_lgb_dataset(pp_valid)
    X_pp_test, lgb_pp_test = make_lgb_dataset(pp_test)
    
    #ハイパーパラメータチューニング
    min_result = np.inf
    for i in range(2,8):
        lgb_params = {
                'objective': "regression",
                'metric':"rmse",
                'num_leaves':2**i-1,
                'seed':2020
        }

        bst = lgb.train(
                    params=lgb_params,
                    train_set=lgb_pp_train,
                    valid_sets=lgb_pp_valid,
                    num_boost_round=1000,
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
        predict = bst.predict(X_pp_valid)
        predict = np.expm1(predict)
        predict = pd.DataFrame(predict,index=valid.index)

        result = output_result(valid,predict,metric='RMSLE')

        #最小値が更新されればbest_paramを更新
        if min_result > result:
            min_result = result
            best_num_leaves = i
            best_iteration = bst.best_iteration
    
    #TrainとValidを結合
    pp_train_valid = pd.concat([pp_train,pp_valid])
    _,lgb_pp_train_valid = make_lgb_dataset(pp_train_valid)

    #もう一度学習
    lgb_params = {
            'objective': "regression",
            'metric':"rmse",
            'num_leaves':2**best_num_leaves-1,
            'seed':2020
    }
    best_bst = lgb.train(params=lgb_params,
                         train_set=lgb_pp_train_valid,
                         num_boost_round=best_iteration)

    #予測を出力
    X_full = pd.concat([X_pp_train,X_pp_valid,X_pp_test])
    predict = best_bst.predict(X_full)
    
    #predictのインデックスを変える
    predict = pd.Series(predict)
    predict.index = list(train.index) + list(valid.index) + list(test.index)
    predict = np.expm1(predict)
    
    return predict

def lgb_modelling_multi(train,valid,test):
    
    """
    LightGBMによる多変量時系列の予測を行います。
    
    Parameters:
    ------
    train,valid,test : rolling_window_valid関数で得られるtrain,valid,testデータ
    
    Return:
    ------
    predict :できる限りの区間予測したものの予測（学習期間もテスト期間も含む）
    
    """
    
    #多変量でも分けたあとに一度フルのデータにして前処理を行う
    full = pd.concat([train,valid,test])

    #対数変換
    full['log_Weekly_Sales'] = np.log(full['Weekly_Sales'])
    del full['Weekly_Sales']
    
    #先に知ることができない情報は1つずらす
    full[full.columns[~full.columns.isin(['log_Weekly_Sales','IsHoliday'])]] = \
        full.drop(['log_Weekly_Sales','IsHoliday'],axis=1).shift()

    #前処理が終わったのでもとに戻す
    pp_train = full[:train.index[-1]]
    pp_valid = full[valid.index[0]:valid.index[-1]]
    pp_test = full[test.index[0]:test.index[-1]]
    
    #LGBMデータセットを作る
    X_pp_train, lgb_pp_train = make_lgb_dataset(pp_train)
    X_pp_valid, lgb_pp_valid = make_lgb_dataset(pp_valid)
    X_pp_test, lgb_pp_test = make_lgb_dataset(pp_test)
    
    lgb_params = {
            'objective': "regression",
            'metric':"rmse",
            'seed':2020
    }
    bst = lgb.train(
                lgb_params,
                train_set=lgb_pp_train,
                valid_sets=lgb_pp_valid,
                num_boost_round=1000,
                early_stopping_rounds=10,
                verbose_eval=False#途中経過を表示しない
            )
    
    #Trainとvalidを結合してデータセットを作る
    pp_train_valid = pd.concat([pp_train,pp_valid])
    _,lgb_pp_train_valid = make_lgb_dataset(pp_train_valid)

    #再度学習
    best_bst = lgb.train(params=lgb_params,
                         train_set=lgb_pp_train_valid,
                         num_boost_round=bst.best_iteration)
    
    #すべての区間の説明変数を結合
    X_full = pd.concat([X_pp_train,X_pp_valid,X_pp_test])

    #予測
    predict = best_bst.predict(X_full)

    #predictのインデックスを変える
    predict = pd.Series(predict)
    predict.index = list(train.index) + list(valid.index) + list(test.index)
    predict = np.exp(predict)
    
    return predict