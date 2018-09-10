#coding:utf-8
'''
Created on 2018年9月6日

@author: xyj
'''

from __future__ import division
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pylab as plt
import LoadData
from sklearn.model_selection import KFold

def getClf():
	return GradientBoostingClassifier(learning_rate=0.05, n_estimators=200, max_depth=7, min_samples_leaf =60, 
               min_samples_split =120, max_features='sqrt', subsample=0.8, random_state=10)


def test():
    # 将已经生成的DataFrame数据读取出来
    df = LoadData.readDataSet()

    df_label = df['label']
    df_features = df.drop(['label', 'id', 'age'], axis=1)

    kf = KFold(n_splits=5)
    print('GBDT model training...')
    predictions = []
    for train_index, val_index in kf.split(df_features):
            
        Train_X = df_features.iloc[train_index]  # 根据训练数据的index得到训练数据
        Train_Y = df_label.iloc[train_index]
        Val_X = df_features.iloc[val_index]


        clf = GradientBoostingClassifier(learning_rate=0.05, n_estimators=200, max_depth=10, min_samples_leaf =60, 
                   min_samples_split =120, max_features='sqrt', subsample=0.8, random_state=10)
       
        clf.fit(Train_X, Train_Y)

        predict_Val_Y = clf.predict(Val_X)
            
        predictions.append(predict_Val_Y)  # 将每一部分的预测结果依次进行添加
        
    predictions = np.concatenate(predictions, axis=0)  # 将[[...],[...]]中的list进行合并，生成一个list

    predict = pd.DataFrame({'label':df_label.values.tolist(), 'pred':list(predictions)}, index=df['id'])
    predict.to_csv('GBDT predictions.csv', index=True, header=True)

    precision = np.count_nonzero(predictions == df_label) / len(predictions)  # 计算预测的正确率
    print(precision, '\n')


if __name__ == '__main__':
    test()