# coding:utf-8

from __future__ import division

from sklearn.ensemble import RandomForestClassifier

import LoadData
import numpy as np


# 随机森林  准确率大于0.93
def getPrecision(X_train, Y_train, X_val, Y_val):
    
    alg = RandomForestClassifier(n_estimators=50, min_samples_split=2, min_samples_leaf=1, oob_score=True)
    
    alg.fit(X_train, Y_train)
    
    # oob模型准确率评估
    print('model oob_score:', alg.oob_score_) 
        
    Y_predict = alg.predict_proba(X_val)[:, 1]  # 参数1表示预测该样本为正类的概率
            
    Y_predict[Y_predict <= 0.44] = 0
    Y_predict[Y_predict > 0.44] = 1
              
    precision = np.count_nonzero(Y_predict == Y_val) / len(X_val)
    
    print('model precision:', precision)
    
#     return precision
    # 获取特征的importance
    featureList = LoadData.getFeatureName('D://tmsc_data/nameListFile.txt')
    
    feature_importances = alg.feature_importances_
    # 将特征名及其重要性分数对应
    Dict = {}
    for (predictor, score) in zip(featureList, feature_importances):
        Dict[predictor] = score
      
    # 对importance值进行排序
    Dict = sorted(Dict.items(), key=lambda d : d[1], reverse=True)
    
    # 打印key-value
#     for key,val in Dict:
#         print key+'\t'+str(val)


#     print '所有的树:%s' % alg.estimators_
 
#     print alg.classes_ # [ 0.  1.]
 
#     print alg.n_classes_ # 2

def getClf():
    
    clf = RandomForestClassifier(n_estimators=500, min_samples_split=2, min_samples_leaf=1, oob_score=True)
    
    return clf

































