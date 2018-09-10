# coding:utf-8

from __future__ import division

from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

import LoadData
import RandomForest
import matplotlib.pyplot as plt

def PlotPrecision_Recall(clf, X, Y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.25)
    
    clf.fit(X_train, y_train)
    
    # precision : array, shape = [n_thresholds + 1]
    # recall : array, shape = [n_thresholds + 1]
    # thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
    precision, recall, thresholds = precision_recall_curve(y_test, clf.predict_proba(X_test)[:, 1])

    print precision, recall, thresholds
    
    precision = precision[:-1]  # Precision values 。the last element is 1.
    recall = recall[:-1]  # 最后一个是0
    
    fig = plt.figure()
    
    ax = fig.add_subplot(1, 1, 1)
    # 设置更加细致的刻度，默认只有5个
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # 画出Precision曲线
    plt.plot(thresholds, precision, label='Precision')
    # 画出Recall曲线
    plt.plot(thresholds, recall, label='Recall')
    # 画出F1_Score曲线
    F1_Score = 2 / ((1 / precision) + (1 / recall))
    plt.plot(thresholds, F1_Score, label='F1_Score')
    # 设置两个轴的边界
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('thresholds')
#     plt.ylabel('precision')
    plt.title('Precision_Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
    
if __name__ == '__main__':
    
    clf = RandomForest.getClf()
    
    df = LoadData.readDataSet()
    
    df_label = df['label']
    df_features = df.drop(['label', 'id'], axis=1)
    
    print "Plotting Precision_Recall Curve..."
    PlotPrecision_Recall(clf, df_features, df_label)
    
    
