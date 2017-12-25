# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:07:29 2017

@author: cisdi
"""


import pandas as pd
import numpy as np
import xlwt
import time
import math
import random
#from otherfunctions import * 
from collections import defaultdict

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from get_features_for_sheet import *
import os 
from training_max_model import MaxEnt
import time

'''
读取保存的w[i]
用最大熵模型预测某一个路径下的所有的文件并逐个输出出来
同时定义了修正规则
'''


def dirlist(path, allfile):  
    '''
    dirlist("/home/yuan/testdir", [])   
    输入路径 (string)
    输出该目录下所有文件的路径 (list)
    '''
    filelist =  os.listdir(path)  

    for filename in filelist:  
        filepath = os.path.join(path, filename)  
        if os.path.isdir(filepath):  
            dirlist(filepath, allfile)  
        else:  
            allfile.append(filepath)  
    return allfile


def rebuild_features(features):
    '''
    将原feature的（a0,a1,a2,a3,a4,...）
    变成 (0_a0,1_a1,2_a2,3_a3,4_a4,...)形式
    '''
    new_features = []
    for feature in features:
        new_feature = []
        for i, f in enumerate(feature):
            new_feature.append(str(i) + '_' + str(f))
        new_features.append(new_feature)
    return new_features


def rebuild_features1(features):
    new_feature = []
    for i , f in enumerate(features):
        new_feature.append(str(i) + '_' + str(f))
    return new_feature


def rule1(d,merged,name):
    '''
    input: 需要检测的预测结果dict , merged_info 
    output: 一样格式的dict
    '''
    
    for x1,x2,y1,y2 in merged: #找所有的合并过的单元格 (48, 49, 34, 46)
        ###操作
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        a = []
        for r in range(x1,x2):
            for c in range(y1,y2):
                a.append(d[r,c])
        ###解析
        if len(set(a)) <= 1: #全部重复 pass
            pass
        else:
            if int(y2) - int (y1) == 1 and (4 in a and 2 in a):#竖排  同时含有4与2
                for r in range(x1,x2):
                    for c in range(y1,y2):
                        d[r,c] = str(d[r,c]) + '-->' + '4'
                        
            elif y2 - y1 == 1 and (1 in a and 2 in a): #竖排  同时含有1与2
                for r in range(x1,x2):
                    for c in range(y1,y2):
                        d[r,c] = str(d[r,c]) + '-->' + '2' 
            elif y2 - y1 == 1 and (2 in a and 5 in a):    #竖排  同时含有5与2
                for r in range(x1,x2):
                    for c in range(y1,y2):
                        d[r,c] = str(d[r,c]) + '-->' + '5'                
      
            elif y2 - y1 == 1 and (4 in a and 5 in a): #竖排  同时含有4与5
                for r in range(x1,x2):
                    for c in range(y1,y2):
                        d[r,c] = str(d[r,c]) + '-->' + '4'          
            
            elif x2-x1 == 1 and y2 - y1 >=5:                     # 横排  取位置中间的那个值
                if d[x1,y1] = d[x1,y2-1]:
                    for r in range(x1,x2):
                        for c in range(y1,y2):
                            d[r,c] = str(d[r,c]) + '-->' + str(d[x1,y1])  
            elif x2-x1 == 1:                     # 横排  取位置中间的那个值
                mid = a[int(len(a)/2)]
                for r in range(x1,x2):
                    for c in range(y1,y2):
                        d[r,c] = str(d[r,c]) + '-->' + str(mid)  
            
            else:
                for r in range(x1,x2):
                    for c in range(y1,y2):
                        d[r,c] = str(d[r,c]) + '-->' + str(d[x1,y1])  
            
                #print(name,'::::::::',x1,x2,y1,y2)

    return d
        

    
        


    
            
            
    

if __name__ == "__main__":
    
    model_path = "./training_results.txt"
    time1 = time.time()
    
    met = MaxEnt()
    raw_data = pd.read_csv('features.csv', header=0)
    data = raw_data.values
    imgs = data[0::, 1::]
    labels = data[::, 0]
    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.001, random_state=23323)
    train_features = rebuild_features(train_features)
    test_features = rebuild_features(test_features)


    
    met.get_model(model_path,train_features, train_labels)
    
    time2 = time.time()
    
    print ('loading model cost', time2 - time1, ' second', '\n')
    
    all_files = dirlist('C:/Users/cisdi/Desktop/test_for_max2',[])#获取文件夹下所有文件的路径
    for files in all_files:

        s = Sheet(files)
        f_map = s.get_features_map()
        results = {} #调用训练好的最大熵模型得出的每个单元格预测类型的结果 后期规则修正则基于此字典
        for i in f_map:
            if f_map[i][0] == 'null':
                
                result = 0
            else:
                result = met.predict1(rebuild_features1(f_map[i]))
            results [i] = result
#        
        
#        for i in f_map:
#            result = met.predict1(rebuild_features1(f_map[i]))
#            results [i] = result
        tx1 = time.time()
        results = rule1(results,s.merged,files)
#        checking(results,s.merged,files)
        
        
#        print('checking cost ', time.time() - tx1)
        
        writer = xlwt.Workbook()
        table = writer.add_sheet('name')
        for i in results:
            table.write(i[0],i[1],results[i])
        name = files.split('\\')[1]
        writer.save('C:/Users/cisdi/Desktop/output_for_max/'+ name +'_results.xls')
        
    time3 = time.time()
    print ('predicting cost ', time3 - time2, ' second', '\n')