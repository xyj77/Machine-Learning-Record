# coding:utf-8
from pandas import DataFrame

import numpy as np
import pandas as pd


# 处理训练数据，去掉数据中的冒号以及特征编号
def processData(filename):
    lines = open(filename, "r").readlines()
    m = len(lines)  # 样本数
    n = len(lines[0].split())  # 　特征数
    data = np.zeros((m, n))
    for i in range(m):
        line = lines[i].split()
        data[i, 0] = line[0]  # label
        for j in range(1, n):  # 特征
            data[i, j] = line[j].split(':')[1]
    return data

# 获取样本ID字段
def getID(filename):
    IdList = []
    lines = open(filename, 'r').readlines()
    for line in lines:
        IdList.append(line)
    IdList = np.array(IdList)
    return IdList

# 获取特征名称
def getFeatureName(filename):
    featurelist = []
    lines = open(filename, 'r').readlines()
    for line in lines:
        featurelist.append(line.strip())
    return featurelist

# 将生成的DataFrame写入excel 防止每次训练的数据不一样
def getDataSet():
    
    ad_filename = './data/adICVFile.txt'
    normal_filename = './data/normalICVFile.txt'
     
    feature_filename = './data/nameListFile.txt'
    
    adIdFilename = './data/adIdFile.txt'
    normalIdFilename = './data/normalIdFile.txt'
    
    ad_data = processData(ad_filename)  # 处理ad类数据
    normal_data = processData(normal_filename)  # 处理normal类数据
    
    adId = getID(adIdFilename)  # 获取ad类数据的ID
    normalId = getID(normalIdFilename)  # 获取normal类数据的ID
    
    ad_data_id = np.column_stack((ad_data, adId))
    normal_data_id = np.column_stack((normal_data, normalId))
    
    featureList = getFeatureName(feature_filename)  # 得到特征名称
    
    data = np.concatenate([ad_data_id, normal_data_id], axis=0)  # 将正类和负类合并到一起
   
    np.random.shuffle(data)  # data被就地按行打乱
    
    featureList.insert(0, 'label')  # 将label放在第一列
    featureList.append('age') #添加年龄
    featureList.append('id') # 将id放在最后一列

    df = DataFrame(data, columns=featureList)
    
    return df
    
def readDataSet():

    df = pd.read_csv('./data/all_shuffle_data_id.csv')
    return df

    

if __name__ == "__main__":

    df = getDataSet()
    df.to_csv('./data/all_shuffle_data_id.csv', index=False, header=True)#写入csv文件