# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:18:50 2017

@author: cisdi
"""

'''
classify csv file by rf model
'''

import pandas as pd
import numpy as np
import xlwt
import time
from collections import defaultdict
#from get_features_for_sheet import *
import os 
from sklearn.ensemble import RandomForestClassifier

from rf_csv_class import *



def classify_csv_by_rf(file1,file2):
    
    time1 = time.time()
    
    raw_data = pd.read_csv('features.csv', header=0)

    f = raw_data.columns[1:]
    train_features = pd.get_dummies(raw_data[f][:])
    train_features.insert(0,'label',raw_data['label'])
    clf = RandomForestClassifier(n_jobs=2)
    
    f = train_features.columns[1:] ######### all features name 
    clf.fit(train_features[f], train_features['label'])
    
    time2 = time.time()
    print ('training cost', time2 - time1, ' second', '\n')
    
    
    #### 每一个map都需要跟r10000拼接 然后取哑特征
    r10000 = raw_data[:][:10000]
    
    ##########################################################################
    

    s = table_csv(file1,file2)
    
    f_map = s.get_features_map_dataframe()
    
    tem_df = pd.concat( [r10000,f_map] )
    
    fff = tem_df.columns[:-2]
    tem_f = pd.get_dummies(tem_df[fff])
    test_df = tem_f[10000:]
    test_df.insert(0,'pos',f_map['pos'])
    
    
    
    results = {} #调用训练好的最大熵模型得出的每个单元格预测类型的结果 后期规则修正则基于此字典
    predict_class = clf.predict(test_df[f])
    
    for i in range (len(predict_class)):
        pos = test_df.loc[i]['pos']
        results [pos] = int (predict_class[i])
        
        
    
#    results = rule1(results,s.merged)        
    writer = xlwt.Workbook()
    table = writer.add_sheet('name')
    for i in results:
        table.write(i[0],i[1],results[i])


    writer.save('C:/Users/cisdi/Desktop/output_for_rf/'+ file1 +'_results.xls')
    
    return ''
        

time0 = time.time()
a = classify_csv_by_rf ('test1.csv' , 'test2.csv')
print('total cost: ', float (time.time()-time0) , 's' )


