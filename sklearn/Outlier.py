# encoding:utf-8

from sklearn.model_selection._split import KFold
from sklearn.preprocessing import scale

import LoadData
import RandomForest
import numpy as np
import warnings
import matplotlib.pyplot as plt


def AgeHist():
    
    df = LoadData.readDataSet()
    df_age = df['Age']
    df_age_normal = df_age[df['label'] == 0]
    
    df_age_normal.hist(bins=40, grid=False).get_figure().savefig('D://tmsc_data/Age_distribution.png')
    
    Dict = {}
    for age in df_age_normal:
        if Dict.has_key(age):
            Dict[age] += 1
        else:
            Dict[age] = 1
     
    Dict = sorted(Dict.items(), key=lambda d : d[0], reverse=False)
     
    keylist = []
    vallist = []
    for key, val in Dict:
        keylist.append(key)
        vallist.append(val)
    print Dict

def AgeSection():
    
    df = LoadData.readDataSet()

    df_features = df.drop(['label','id'], axis=1)
    df_features = df_features[df['label'] == 0]
    
    df_age56to59 = df_features[df['Age'] <= 59]  # 10 rows
    df_age56to59 = df_age56to59.drop('Age', axis=1)
    
    df_age60to70 = df_features[df['Age'] <= 70]  # 435 rows
    df_age60to70 = df_age60to70[df['Age'] >= 60]
    df_age60to70 = df_age60to70.drop('Age', axis=1)
    
    df_age71to82 = df_features[df['Age'] <= 82]  # 2092 rows
    df_age71to82 = df_age71to82[df['Age'] >= 71]
    df_age71to82 = df_age71to82.drop('Age', axis=1)
    
    df_age83to90 = df_features[df['Age'] >= 83]  # 515 rows
    df_age83to90 = df_age83to90[df['Age'] <= 90]
    df_age83to90 = df_age83to90.drop('Age', axis=1) 
    
    df_age91to96 = df_features[df['Age'] >= 91]  # 24 rows
    df_age91to96 = df_age91to96.drop('Age', axis=1) 
    
    return df_age56to59, df_age60to70, df_age71to82, df_age83to90, df_age91to96


    

if __name__ == '__main__':
    
    warnings.filterwarnings("ignore")
    
    df_age56to59, df_age60to70, df_age71to82, df_age83to90, df_age91to96 = AgeSection()
    
    df_list = [df_age56to59, df_age60to70, df_age71to82, df_age83to90, df_age91to96]
    
#     df_list = [df_age56to59]


    for df in df_list:
        df_index = np.array(df.index)
        df = np.array(df)
        scaled_df = scale(df)
        avg_vec = np.mean(scaled_df, axis=0)
        dot_list = []
        for i in range(len(df)):
            sub = df[i] - avg_vec
            dot_list.append(np.dot(sub, sub))
        dot_list = np.sqrt(dot_list)
        
        Dict = {}
        for (index, score) in zip(df_index, dot_list):
            Dict[index] = score
        Dict = sorted(Dict.items(), key=lambda d : d[1], reverse=True)
        for key, val in Dict:
            print key, val
        print'11111111111111111111111111111'

        plt.figure()
    
        plt.scatter(df_index, dot_list)
       
        plt.xlim([0, 4600])
        plt.ylim([1.5, 4.0])
        
        plt.xlabel('Samples_ID')
        plt.ylabel('Distance')
        plt.title('Out lier normal')
        
        plt.show()

















