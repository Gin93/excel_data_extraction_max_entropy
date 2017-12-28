# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:48:26 2017

@author: cisdi
"""

import copy

from collections import Counter 

import csv

import numpy as np
import pandas as pd 

def fuck_the_nan(l):
    '''
    输入一个list
    将里面的nan替换为''
    '''
    ll = []
    for i, ele in enumerate(l):
        if str(ele) == 'nan':
            ll.append('')
        else:
            ll.append(ele)
    return ll

def op_splitting(csv_data,info):
    '''
    根据split_info分割表格，此时的表格只有纵向合并的表格
    '''

    output = []
#    with open(csv_data,"r",encoding='utf8') as csvfile: #打开原始文件 
        
    df=pd.read_csv(csv_data, header= None, encoding="utf-8")  #读取csv文件
    df.fillna(value='')
    
    r  = np.array(df)  #转化成ndarray
    
    csvID = 0
    rows = 0
    for eachtable in info:
        output_csv_name = csv_data.replace('.csv','') + '_'+ str(csvID) + '.csv'
        output.append(output_csv_name)
        with open(output_csv_name ,"w",encoding='utf-8',newline='' ) as outputcsv:
            writer = csv.writer(outputcsv)
            for label in eachtable:
                data = list(r[rows])
                data1 = fuck_the_nan(data)
                data1 = [label] + data1
                writer.writerow(data1)
                rows += 1
            csvID += 1
    return output


def split_info(l):
    '''
    输入为一个一维list [label1,l2,l2,l3]
    输出为拆分过后的表,一个二维数组 即[[l1,l2,l3],[l4,l5,l6]] l1,l2,l3 是一个表, 其余三个为另一个表
    '''
    output = []
    tem_list = []    
    
    max = 0
    for i in l:
        b = int(i)
        
        if b >= max:
            tem_list.append(i)
            max = b
        else: #出现第二个表，把第一个表存起来,清空list,并且存入当前的label
            output.append(tem_list)
            tem_list = []
            tem_list.append(i)
            max = 0
    output.append(tem_list)
    
    if len (output)> 1:
        if '4' not in output[-1]:  #可以做一波修改
            print('final_check_issues')
            #设置为最后一个表的最后一个分类结果 4 or 5
            last = output[-2][-1]
            last = '5'
            for a1 in output[-1]:
                output[-2].append(last)
            output.pop(-1)

    
    return output   




def split_table(raw_table,rows,cols,raw_file):
    '''
    传入分类结果（单元格粒度），行数与列数，文件路径
    '''
    
    
    def split_table_empty_col(raw_table,rows,cols):
        '''
        分割情况1
        空列可以作为分割依据
        '''
        col_split = []
        for c in range(cols):
            if sum(raw_table[:,c]) == 0:
                col_split.append(c)
        ###分析col_split 为空则说明不用分割
        if col_split == []:
            return []
        ### 非空时，分析是否是有效的分割线
        l = copy.deepcopy(col_split)
        for i in col_split:
            if i in( 0,1,cols-1): # 明显不靠谱的几个值
                l.remove(i)
                continue
            if i+1 in col_split:
                l.remove(i)      
        return l
    
    def split_table_row(raw_table,rows,cols):
        '''
        分割情况3 // 或可以作为最终分割器
        针对每行数据都属于同一个table
        有多个table纵向叠在一起的情况
        输出为拆分过后的表,一个二维数组 即[[l1,l2,l3],[l4,l5,l6]] 
        '''
        row_label_list = [] 
        for r in range(rows): 
            row_data = raw_table[r,:]
            majority = Counter(row_data).most_common(1)[0][0]
            if majority == 0:
                majority = Counter(row_data).most_common(2)[1][0]
            row_label_list.append(majority)
        line = split_info(row_label_list)
                
        files_path = op_splitting(raw_file,line)
        return files_path

    '''
    x1 = split_table_empty_col(raw_table,rows,cols)
    
    if x1:
        return x1
    '''
    
    x2 = split_table_row(raw_table,rows,cols)
    
    return x2





