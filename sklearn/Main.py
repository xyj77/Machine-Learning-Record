# coding:utf-8
from __future__ import division

from sklearn.model_selection import KFold

import LoadData
import RandomForest
import SVMClassfier
import GBDT
import Xgboost
import numpy as np
import pandas as pd
import xgboost as xgb

pd.set_option('display.width', 500)

def train(clf, df, name=''):
    df_label = df['label']
    # df_features = df.drop(['label', 'id', 'age'], axis=1) #without age
    df_features = df.drop(['label', 'id'], axis=1)        #keep age
    kf = KFold(n_splits=5)
    print(name + ' model training...')
    predictions = []
    for train_index, val_index in kf.split(df_features):
        
        Train_X = df_features.iloc[train_index]  # 根据训练数据的index得到训练数据
        Train_Y = df_label.iloc[train_index]
        Val_X = df_features.iloc[val_index]
        
        clf.fit(Train_X, Train_Y)

        if name == 'RandomForest':
            predict_Val_Y = clf.predict_proba(Val_X)[:, 1]  # 1表示样本为1的概率
            
            predict_Val_Y[predict_Val_Y <= 0.44] = 0  # 概率阈值设置为0.44
            predict_Val_Y[predict_Val_Y > 0.44] = 1

        else:
            predict_Val_Y = clf.predict(Val_X)
        
        predictions.append(predict_Val_Y)  # 将每一部分的预测结果依次进行添加
    
    predictions = np.concatenate(predictions, axis=0)  # 将[[...],[...]]中的list进行合并，生成一个list

    predict = pd.DataFrame({'label':df_label.values.tolist(), 'pred':list(predictions)}, index=df['id'])
    predict.to_csv(name + 'predictions.csv', index=True, header=True)

    precision = np.count_nonzero(predictions == df_label) / len(predictions)  # 计算预测的正确率
    print(precision, '\n')


if __name__ == '__main__':
    
    # # 生成DataFrame数据  运行一次即可
    # df = LoadData.getDataSet()
    
    # 将已经生成的DataFrame数据读取出来
    df = LoadData.readDataSet()
    
    clf = RandomForest.getClf()
    train(clf, df, name='RandomForest')
    
    clf = SVMClassfier.getClf()
    train(clf, df, name='SVM')

    clf = GBDT.getClf()
    train(clf, df, name='GBDT')

    #xgboost 框架直接输入数据训练，不是先获取分类器
    clf = Xgboost.train()