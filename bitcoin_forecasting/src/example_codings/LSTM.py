import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size,)

        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def lstm_preprocessing(train,valid):
    
    """
    LSTM用に前処理（主に正規化）を行います
    
    Returns:
    ------
    train_inout,valid_inout : 正規化を行い、LSTM用に作成したデータセット
    scaler : 逆変換用のsklearnの正規化モジュール
    
    """
    
    #正規化
    scaler = MinMaxScaler(feature_range=(-1,1))
    train_normalized = scaler.fit_transform(train.values.reshape(-1,1))
    valid_normalized = scaler.transform(valid.values.reshape(-1,1))
    
    #結合
    concat_data = np.concatenate((train_normalized,valid_normalized))
    
    #tensor化
    concat_data = torch.FloatTensor(concat_data).view(-1)
    
    #データセット作成
    tw = 12
    L = len(concat_data)
    inout = []
    for i in range(L - tw):
        seq = concat_data[i:i+tw]
        label = concat_data[i+tw]
        inout.append((seq,label))
        
    #分割
    train_inout = inout[:-len(valid)]
    valid_inout = inout[-len(valid):]

    return train_inout,valid_inout,scaler


def lstm_modelling(train, valid, test):
    
    """
    LSTMによる単変量の予測を行います。early_stoppingを行います
    
    Parameters:
    ------
    train,valid,test : rolling_window_valid関数で得られるtrain,valid,testデータ
    
    Return:
    ------
    predict :できる限りの区間予測したものの予測（学習期間もテスト期間も含む）
    
    """
    
    #trainとvalidの前処理
    train_inout,valid_inout,scaler = lstm_preprocessing(train,valid)
    
    #train_validとtestの前処理
    train_valid = pd.concat([train,valid])
    train_valid_inout, test_inout, scaler_tv = lstm_preprocessing(train_valid,test)
    
    #初期設定
    epochs = 100            #最大epoch回数
    early_stopping_num = 10 #何回精度が向上しなければ学習をやめるか
    valid_loss_min = np.inf
    count = 0               #early_stoppingのためのカウンター
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #ハイパラチューニング用学習
    train_size = len(train_inout)
    valid_size = len(valid_inout)

    for i in range(epochs):
        # trainで学習
        model.train()
        train_loss = 0
        for seq, label in train_inout:
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred, label)
            train_loss += single_loss
            single_loss.backward()
            optimizer.step()

        # valid区間で予測
        model.eval()
        valid_loss = 0
        for seq, label in valid_inout:
            y_pred_valid = model(seq)
            valid_loss += loss_function(y_pred_valid, label)

        # validの合計lossが前回より減少している場合
        if valid_loss < valid_loss_min:
            count = 0                    # countの初期化
            valid_loss_min = valid_loss  # loss最小値の更新
            best_model = model           #ベストなモデルを保存

        # 増加している場合
        else:
            count += 1
            if count >= early_stopping_num: # 回数が上限に達した場合
                break

    best_epochs = i - count
    
    # 予測
    predict = []
    best_model.eval()
    for seq, label in train_valid_inout:
        y_pred = best_model(seq)
        predict.append(y_pred.item())

    for seq, label in test_inout:
        y_pred = best_model(seq)
        predict.append(y_pred.item())

    predict = scaler_tv.inverse_transform(np.array(predict).reshape(-1,1))
    predict = pd.DataFrame(predict)
    predict.index = list(train.index[len(train_inout[0][0]):]) + list(valid.index) + list(test.index)

    return predict