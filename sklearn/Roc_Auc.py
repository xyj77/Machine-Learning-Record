# coding:utf-8
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

import LoadData
import RandomForest
import matplotlib.pyplot as plt


def PlotRocAuc(clf, X, Y):
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.25)
    
    clf.fit(X_train, y_train)
    
    # Determine the false positive and true positive rates
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    
#     print fpr, tpr, thresholds
    
    
    # Calculate the AUC
    roc_auc = auc(fpr, tpr)
    print 'ROC AUC: %0.2f' % roc_auc
    
    # Plot of a ROC curve for a specific class
    plt.figure()
    
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    
    plt.plot([0, 1], [0, 1], 'k--')
   
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    
if __name__ == '__main__':
    # get RandomForest model
    clf = RandomForest.getClf()
    # read data from file
    df = LoadData.readDataSet()
    # split label from features
    df_label = df['label']
    df_features = df.drop(['label','id'], axis=1)
    
    print "Plotting ROC Curve..."
    PlotRocAuc(clf, df_features, df_label)
    
    
    
    
