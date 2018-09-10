# coding:utf-8
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

import LoadData
import matplotlib.pyplot as plt
import numpy as np


# assume classifier and training data is prepared...
def PlotLearningCurve(clf, X, Y):
    train_sizes, train_scores, test_scores = learning_curve(
            clf, X, Y, cv=10, n_jobs=-1, train_sizes=np.linspace(.1, 1., 10), verbose=0) 
    
    
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.title("RandomForestClassifier")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.ylim((0.7, 1.01))
    plt.grid()
    
    # Plot the average training and test score lines at each training set size
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Test score")
    plt.legend(loc="best")
    
    # Plot the std deviation as a transparent range at each training set size
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
#                     alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                     alpha=0.1, color="r")
    
    
    # Draw the plot and reset the y-axis
    plt.draw()
    plt.show()

    
    
    
if __name__ == '__main__':
    
    clf_RF = RandomForestClassifier(n_estimators=50, min_samples_split=2, min_samples_leaf=1, oob_score=True)
    clf_SVC = clf = SVC(C=10, kernel='rbf', gamma=30)
    
    
    df = LoadData.readDataSet()
    
    df_label = df['label']
    df_features = df.drop(['label','id'], axis=1)
    
    
    PlotLearningCurve(clf_RF, df_features, df_label)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
