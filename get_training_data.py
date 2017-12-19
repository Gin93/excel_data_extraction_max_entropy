# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:27:50 2017

@author: cisdi
"""
import xlrd
import string
import pandas as pd 
import csv
import os
import random
from get_features_for_sheet import *
import time

'''
加载csv标记文件
逐个文件读取并且new成sheet类型
对每一个点用sheet里的方法得到该位置的特征
根据标记信息 加上label
再append
再输出保存
'''

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



a = read_csv('dataset.csv')#读取csv文件

data_set_path = 'C:/Users/cisdi/Desktop/data_for_max'
#data_set_path = 'C:/Users/cisdi/Desktop/test'

time1 = time.time()

features = []    
for file,x1,y1,x2,y2,label in a:        
    file_name = os.path.join(data_set_path,file)
    try:
        sheet = Sheet(file_name) 
        for row in range(int(x1),int(x2)+1,1):
            for col in range(int(y1),int(y2)+1,1):                
                f = sheet.get_features((row-1,col-1)) #邻居范围在class里面定义        
                features.append( [label] + f )
    except FileNotFoundError:
        print(file_name)

        
time2 = time.time()

print ('getting features cost', time2 - time1, ' second', '\n')


l = len(features[0])
header = ['label']
f_count = 1
for i in range(l-1):
    header = header + ['f'+str(f_count)]
    f_count +=1


with open ('features.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerows([header])
    
    read_data = False
    count = 0
    for i in features :
        if i [1] != 'null':
            if i [0] == '4':
                
                if random.randint(0,100) < 15:
                    writer.writerows([i])
            else:
                if i [0] == '1':
                    writer.writerows([i])
                    writer.writerows([i])
                if i [0] == '5':
                    writer.writerows([i])
                    writer.writerows([i])
                else:
                    writer.writerows([i])

time3 = time.time()
print ('writing features to csv cost', time3 - time2, ' second', '\n')    