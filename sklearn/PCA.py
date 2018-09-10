# coding:utf-8
from __future__ import division

from sklearn.decomposition import PCA
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.model_selection import KFold

import LoadData
import numpy as np
import pandas as pd


def getPCAData():

    df = LoadData.readDataSet()

    df_features = df.drop('label', axis=1)
    df_label = df['label']
    
    pca = PCA(n_components=350, svd_solver='full', copy=True)  # 保留350个特征
    
    X_pca = pca.fit_transform(df_features)
    
    X_pca = pd.DataFrame(X_pca)
    
    return X_pca, df_label
    
#     print pca.components_

#     df_pca = DataFrame(pca.get_covariance(),index=featureList,columns=featureList)
     
#     df_pca.to_excel('D://tmsc_data/pca_covariance.xlsx', sheet_name='Sheet1')

if __name__ == '__main__':
    
    df_features, df_label = getPCAData()
    
    kf = KFold(n_splits=10)
    predictions = []
    
    print('PCA with RandomForest model training...')
    
    for train_index, val_index in kf.split(df_features):
        
        Train_X = df_features.iloc[train_index]
        Train_Y = df_label.iloc[train_index]
        Val_X = df_features.iloc[val_index]
        
        clf = RandomForestClassifier(n_estimators=50, min_samples_split=2, min_samples_leaf=1, oob_score=True)
        clf.fit(Train_X, Train_Y)
        
        predict_Val_Y = clf.predict_proba(Val_X)[:, 1]
        
        predict_Val_Y[predict_Val_Y <= 0.44] = 0
        predict_Val_Y[predict_Val_Y > 0.44] = 1
        
        predictions.append(predict_Val_Y)
    
    predictions = np.concatenate(predictions, axis=0)
    
    precision = np.count_nonzero(predictions == df_label) / len(predictions)
    
    print(precision)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
