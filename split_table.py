# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 13:57:24 2017

@author: cisdi
"""

'''
search
'''
import numpy as np
from collections import Counter

def split1(l):
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


def final_check(l):
    '''
    对分割后的最后一个表做处理判断其是否是分类错误
    输入为一个二维list
    True为分类错误，需要进行合并
    '''
    for i,j in l:
        if j == '4':
            return False 
    return True        
    

def split_table(results,rows,cols):
    '''
    (dict{}) --> [表1[(),(),(),()],表2[(),(),(),()]]
    '''
    a = np.empty([rows,cols], dtype =  int)
    
    for i in results:
        a[i[0],i[1]] = results[i] ##转化为容易分析的ndarray
    
    line = []
    for r in range(rows):
        left = 0
        for c in range(cols):
            if left == 0:
                if a[r][c] != 0:
                    left = a[r][c]
            else:
                if a[r][c] != 0:
                    if a[r][c] != left and 5 not in (a[r][c],left):
                        line.append((r,c))
                        left = a[r][c]
                
                
            
    return [line,a]

def split_table_empty_col(results,rows,cols):
    a = np.empty([rows,cols], dtype =  int)
    
    for i in results:
        a[i[0],i[1]] = results[i] ##转化为容易分析的ndarray
    
    col_split = []
    for c in range(cols):
        if sum(a[:,c]) == 0:
            line.append(c)
            
    return col_split


def split_table_row(results,rows,cols):
    a = np.empty([rows,cols], dtype =  int)
    
    for i in results:
        a[i[0],i[1]] = results[i] ##转化为容易分析的ndarray


    l = []
    for r in range(rows): # 取第三列的数据分析 暂时
        l.append(a[r,2])
    print(l)
    line = split1(l)
            
    return line
    