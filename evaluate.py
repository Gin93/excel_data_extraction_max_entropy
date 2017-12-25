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




def checking (csv_data,file_name,r,l,result):

    for file,x1,y1,x2,y2,label in csv_data: 

        if file == os.path.basename(file_name):  # 这样只检查标记过的 未标记过的也不会报错
            if r+1 >= int(x1) and r+1<= int(x2) and l+1 >= int(y1) and l+1 <= int(y2):
                if str(result)[-1] != '0':
                    return (str(result)[-1], str(label))
                
                    if str(result)[-1] != str(label):
                        print(file_name)
                        print((r,l))
                        print('result is' , result, 'but shuold be', label)

def checking1 (csv_data,file_name,r,l,result):

    for file,x1,y1,x2,y2,label in csv_data: 

        if file == os.path.basename(file_name):  # 这样只检查标记过的 未标记过的也不会报错
            if r+1 >= int(x1) and r+1<= int(x2) and l+1 >= int(y1) and l+1 <= int(y2):
                if str(result)[-1] != '0':
                
                    if str(result)[-1] != str(label):
                        print(file_name)
                        print((r,l))
                        print('result is' , result, 'but shuold be', label)
                        return (False,str(result),str(label))
    return(True,str(result),str(label))
        
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
    
    
    
    actual = []
    predict = []
    classfication_results = []
    all_files = dirlist('C:/Users/cisdi/Desktop/test_for_max',[])#获取文件夹下所有文件的路径
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
        
        results = rule1(results,s.merged,files)
        
        
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



                 
        writer = xlwt.Workbook()
        table = writer.add_sheet('name')
        for i in results:
            table.write(i[0],i[1],results[i])
        name = files.split('\\')[1]
        writer.save('C:/Users/cisdi/Desktop/output_for_max/'+ name +'_results.xls')
        
    time3 = time.time()
    print ('predicting cost ', time3 - time2, ' second', '\n')
    
    
    confusion_matrix_plot_matplotlib(actual, predict)
    df=pandas.DataFrame(classfication_results,columns=['actual','predict']) 
    correct=df[df.actual==df.predict]
    for i in ('1','2','4','5'):
        R=sum(correct.predict==i)/sum(df.actual==i)
        P=sum(correct.predict==i)/sum(df.predict==i)
        F=R*P*2/(R+P)
        print(i,':\n','R=',R,' P=',P,' F=',F)
    