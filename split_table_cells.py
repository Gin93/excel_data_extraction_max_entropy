# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:48:26 2017

@author: cisdi
"""

import copy

def split_table(raw_table,rows,cols):
    
    
    def split_table_empty_col(raw_table,rows,cols):
        '''
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
    
    return split_table_empty_col(raw_table,rows,cols)
    

def one_table(results,rows,cols):
    '''
    输入为ndarray 输出为bool
    根据单元格的分类结果显示是否只有一个表
    '''
    
    return ''



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
    