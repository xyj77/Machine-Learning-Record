# encoding:utf-8

from pandas import DataFrame

import numpy as np
import pandas as pd


def writeToFile(features,filename):
    f = open(filename, 'w')
    for feature in features:
        f.write(feature)
        f.write('\n')
    f.close()


def reduceFeature():
    
    #将已经生成的DataFrame数据读取出来
    df = pd.read_excel('D://tmsc_data/all_shuffle_data.csv', sheet_name='Sheet1') 
     
    df_features = df.drop(['label','id'], axis=1)
     
    # 计算斯皮尔曼相关系数（矩阵） 时间略长
    df_corr = df_features.corr(method='spearman')
     
    df_corr = DataFrame(df_corr)
     
    #将相关系数矩阵写入excel表格
    df_corr.to_excel('D://tmsc_data/corr.xlsx',sheet_name='Sheet1')
    
#     df_corr = pd.read_excel('D://tmsc_data/corr.xlsx', sheet_name='Sheet1')
    
    # 将对角线变为0
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr
       
    drops = []  # 需要丢掉的特征
    keeps = []  # 需要保留的特征
    # 循环
    for col in df_corr.columns.values:
        # 如果当前特征已经是之前要丢掉的特征，不再重复加入，继续判断下一个    
        if np.in1d([col], drops):
            continue
        # 找出高相关的变量
        corr = df_corr[abs(df_corr[col]) > 0.98].index
        corr_array = np.union1d([], corr)
        keeps.append(col)
        drops = np.union1d(drops, corr)
        print col, corr_array # 特征及其相关特征
    
#     print "nDropping", drops.shape[0], "highly correlated features...n", drops
   
    # 将保留/丢弃的特征写入文件
    keep_filename = 'D://tmsc_data/keep_features.txt'
    drop_filename = 'D://tmsc_data/drop_features.txt'
    
    writeToFile(keeps,keep_filename)
    writeToFile(drops,drop_filename)
   
   
if __name__ == '__main__':
    
    # 通过Spearman系数 去掉一些相关性强的特征  运行一次得到结果即可
    reduceFeature()  # 将保留的特征写入文件
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
