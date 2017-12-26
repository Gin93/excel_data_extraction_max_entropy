# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:44:50 2017

@author: cisdi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 08:43:59 2017

@author: cisdi
"""

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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np





'''
用于测试调参评估训练好的最大熵模型
输出分类结果,
结合了csv文件，对分类错误的文件进行了分类
'''




def confusion_matrix_plot_matplotlib(y_truth, y_predict, cmap=plt.cm.Blues):
    """Matplotlib绘制混淆矩阵图
    parameters
    ----------
        y_truth: 真实的y的值, 1d array
        y_predict: 预测的y的值, 1d array
        cmap: 画混淆矩阵图的配色风格, 使用cm.Blues
    """
    cm = confusion_matrix(y_truth, y_predict)
    plt.matshow(cm, cmap=cmap)  # 混淆矩阵图
    plt.colorbar()  # 颜色标签
     
    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
     
    plt.xlabel('True label')  # 坐标轴标签
    plt.ylabel('Predicted label')  # 坐标轴标签
    plt.show()  # 显示作图结果
    
    
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
                try:
                    a.append(d[r,c])
                except:
                    a.append(0)
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
                
                if d[x1,y1] == d[x1,y2-1]:  # 对于长度大于5的如果首尾相同，则取这个值
                    xxx = d[x1,y1]
                    for r in range(x1,x2):
                        for c in range(y1,y2):
                            d[r,c] = str(d[r,c]) + '-->' + str(xxx)  
                else:  #有待商议
                    mid = a[int(len(a)/2)]
                    for r in range(x1,x2):
                        for c in range(y1,y2):
                            d[r,c] = str(d[r,c]) + '-->' + str(mid)    
            elif x2-x1 == 1:                # 横排  取位置中间的那个值
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
        
def read_csv(csv_file_name):
    '''
    input: csv_file_name
    ouput: excel_filename, x1,y1,x2,y2,label 
    '''
    o = []
    with open (csv_file_name) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            o.append(row)  
    return o 

def find_untrained_data(csv_data,file_name):
    '''
    return bool
    '''
    a = set()
    for file,x1,y1,x2,y2,label in csv_data: 
        a.add(file)
        
    return os.path.basename(file_name) in a
        


def checking (csv_data,file_name,r,l,result):

    for file,x1,y1,x2,y2,label in csv_data: 

        if file == os.path.basename(file_name):  # 这样只检查标记过的 未标记过的也不会报错
            if r+1 >= int(x1) and r+1<= int(x2) and l+1 >= int(y1) and l+1 <= int(y2):
                if str(result)[-1] != '0':
                    return (str(result)[-1], str(label))
                

def checking1 (csv_data,file_name,r,l,result):

    for file,x1,y1,x2,y2,label in csv_data: 

        if file == os.path.basename(file_name):  # 这样只检查标记过的 未标记过的也不会报错
            if r+1 >= int(x1) and r+1<= int(x2) and l+1 >= int(y1) and l+1 <= int(y2):
                if str(result)[-1] != '0':
                
                    if str(result)[-1] != str(label):
                        '''
                        print(file_name)
                        print((r,l))
                        print('result is' , result, 'but shuold be', label)
                        '''
                        return (False,str(result),str(label))
    return(True,str(result),str(label))

time0 = time.time()
if __name__ == "__main__":
    

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
    
    
    actual = []
    predict = []
    classfication_results = []
    all_files = dirlist('C:/Users/cisdi/Desktop/test_for_max2',[])#获取文件夹下所有文件的路径
    
    for files in all_files:
        print('start classify:' , files)

        s = Sheet(files)
        f_map = s.get_features_map_dataframe()
        
        tem_df = pd.concat( [r10000,f_map] )
        
        fff = tem_df.columns[:-2]
        tem_f = pd.get_dummies(tem_df[fff])
        test_df = tem_f[10000:]
        test_df.insert(0,'pos',f_map['pos'])
        
        rows = test_df.iloc[:,0].size
        results = {} #调用训练好的最大熵模型得出的每个单元格预测类型的结果 后期规则修正则基于此字典
        time3 = time.time()
#        print ('strat predict', time3 - time2, ' second', '\n')
        
        
        predict_class = clf.predict(test_df[f])
        
        for i in range (len(predict_class)):
            pos = test_df.loc[i]['pos']
            results [pos] = int (predict_class[i])
            
#        for i in range(rows):
#            
#            data = test_df.loc[i]
#            result = int (clf.predict(data[f].reshape(1,-1)))
#            pos = data['pos']
#            results [pos] = result
            
            
            
            
        time4 = time.time()
#        print ('predict cost ', rows , time4 - time3, ' second', '\n')
#
        
        results = rule1(results,s.merged,files)        
        time5 = time.time()
#        print ('rule1_check cost ' , time5 - time4, ' second', '\n')


        csv_data = read_csv('dataset.csv')#读取csv文件
        tx1 = time.time()
        
        for pre_result in results:
            try:
                (xxx,yyy) = checking(csv_data,files,pre_result[0],pre_result[1],results[pre_result])
                actual.append(yyy)
                predict.append(xxx)
                classfication_results.append([yyy,xxx])
            except TypeError:
                pass
            
        for pre_result in results:

            o1,o2,o3 = checking1(csv_data,files,pre_result[0],pre_result[1],results[pre_result])
            if not o1:
                results [pre_result] = ('wrong, ''result is: ' , o2, ' but shuold be: ', o3)
        

#        ################只输出错误的
#        flag = False 
#        
#        for i in results:
#            if 'wrong' in str(results[i]):
#                flag = True
                
        trained = False
        trained = find_untrained_data(csv_data,files)
        
        flag = True
        if flag:
            writer = xlwt.Workbook()
            table = writer.add_sheet('name')
            for i in results:
                table.write(i[0],i[1],results[i])
            name = files.split('\\')[1]
            if not trained:
                name = 'untrained' + name
            writer.save('C:/Users/cisdi/Desktop/output_for_max/'+ name +'_results.xls')
        
#    time3 = time.time()
#    print ('predicting cost ', time3 - time2, ' second', '\n')
#    
#    
    confusion_matrix_plot_matplotlib(actual, predict)
    df=pandas.DataFrame(classfication_results,columns=['actual','predict']) 
    correct=df[df.actual==df.predict]
    for i in ('1','2','4','5'):
        R=sum(correct.predict==i)/sum(df.actual==i)
        P=sum(correct.predict==i)/sum(df.predict==i)
        F=R*P*2/(R+P)
        print(i,':\n','R=',R,' P=',P,' F=',F)

    print('total cost: ', time.time()-time0 )