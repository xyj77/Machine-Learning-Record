#coding:utf-8
from __future__ import division

from sklearn.svm import SVC

import numpy as np


def getSVMPrecision(X_train,Y_train,X_val,Y_val):
    
    #支持向量分类  准确率大约为0.93
    clf = SVC(C=10, kernel='rbf', gamma=30)
             
    clf.fit(X_train, Y_train)
            
    Y_predict = clf.predict(X_val)
             
    acc = np.count_nonzero(Y_predict == Y_val) / len(X_val)
             
    return acc


def getClf():
	return SVC(C=10, kernel='rbf', gamma=30)
    