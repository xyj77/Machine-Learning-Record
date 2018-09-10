#coding:utf-8
'''
Created on 2018年9月6日

@author: xyj
'''

from __future__ import division
import pandas as pd
import numpy as np
import xgboost as xgb
import LoadData
from sklearn.model_selection import KFold


def train():
	# 将已经生成的DataFrame数据读取出来
	df = LoadData.readDataSet()

	df_label = df['label']
	# df_features = df.drop(['label', 'id', 'age'], axis=1) #without age
	df_features = df.drop(['label', 'id'], axis=1)        #keep age

	kf = KFold(n_splits=5)
	print('Xgboost model training...')
	predictions = []
	for train_index, val_index in kf.split(df_features):
		Train_X = df_features.iloc[train_index]  # 根据训练数据的index得到训练数据
		Train_Y = df_label.iloc[train_index]
		Val_X = df_features.iloc[val_index]
		dtrain = xgb.DMatrix(Train_X, Train_Y)
		dtest = xgb.DMatrix(Val_X)

		num_round = 200
		params = {
		    'booster': 'gbtree',            #分类器：基于树的模型
		    'objective': 'binary:logistic', #损失函数：二分类交叉熵
		    'subsample': 0.8,       #采样率，和GBM中的subsample参数一模一样。这个参数控制对于每棵树，随机采样的比例。减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。典型值：0.5-1
		    'colsample_bytree': 0.8,#和GBM里面的max_features参数类似。用来控制每棵随机采样的列数的占比(每一列是一个特征)。典型值：0.5-1
		    'eta': 0.05,      #和GBM中的 learning rate 参数类似，通过减少每一步的权重，可以提高模型的鲁棒性。典型值为0.01-0.2。
		    'max_depth': 7,   #
		    'gamma': 0.0,     #在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的。
		    'silent': 1,      #不打印训练过程
		    'eval_metric':'error', #对于有效数据的度量方法。对于回归问题，默认值是rmse，对于分类问题，默认值是error
		    'lambda':1,       #[默认1]权重的L2正则化项。(和Ridge regression类似)。这个参数是用来控制XGBoost的正则化部分的。
            'alpha':1         #[默认1]权重的L1正则化项。(和Lasso regression类似)。可以应用在很高维度的情况下，使得算法的速度更快。
		    }      

		clf = xgb.train(params, dtrain, num_round)
		predict_prob = clf.predict(dtest)
		predict_Val_Y = [x>=0.5 for x in predict_prob]

		predictions.append(predict_Val_Y)  # 将每一部分的预测结果依次进行添加
	    
	predictions = np.concatenate(predictions, axis=0)  # 将[[...],[...]]中的list进行合并，生成一个list

	predict = pd.DataFrame({'label':df_label.values.tolist(), 'pred':list(predictions)}, index=df['id'])
	predict.to_csv('Xgboost predictions.csv', index=True, header=True)

	precision = np.count_nonzero(predictions == df_label) / len(predictions)  # 计算预测的正确率
	print(precision, '\n')

if __name__ == '__main__':
    train()