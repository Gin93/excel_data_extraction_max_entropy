# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:43:04 2017

@author: cisdi
"""

import xlrd

import string
import pandas as pd 
import csv
import os
import random

class Sheet:
    def __init__(self, file_path):
        
        try:
            self.sheet = xlrd.open_workbook(file_path, formatting_info=True).sheets()[0]
        except:
            self.sheet = xlrd.open_workbook(file_path).sheets()[0]
            
        self.rows = self.sheet.nrows
        self.cols = self.sheet.ncols
        self.merged = self.sheet.merged_cells
        self.features = [] 
#        self.neighbors = ['up','down','left','right','up2','down2','left2','right2','upl','upr','downl','downr']
        self.neighbors = ['up','down','left','right','up2','down2','left2','right2']
       
        
    def filling(self):
        '''
        将合并过的单元格全部填充
        可以假填充:
            if merged:
                data = cell_value(x1,y1)
        '''
        
    def selflens(self,data):
#        l = len(str(data))
#        if l <=3:
#            return 'datashort'
#        elif l <= 6:
#            return 'datanormallen'
#        elif l <= 9:
#            return 'datalong'
#        elif l <= 15:
#            return 'dataverylong'
#        else:
#            return 'dataextremelylong'
        
        count_en = count_dg = count_sp = count_zh = count_pu = 0

        for c in str(data):
            if c in string.ascii_letters:
                count_en += 1
            elif c.isdigit():
                count_dg += 1
            elif c.isspace():
                count_sp += 1
            elif c.isalpha():
                count_zh += 1
            else:
                count_pu += 1
        l = count_en + count_zh + count_pu
        if l < 1:
            return 'dataveryshort'
        elif l == 1:
            return 'datashort'
        elif l <= 6:
            return 'datanormallen'
        elif l <= 12:
            return 'datalong'
        elif l <= 20:
            return 'dataverylong'
        else:
            return 'dataextremelylong'        
    

    
    def valid(self,x,y,rows,cols): # 可以优化
        if x < 0 or y < 0 or x > rows-1 or y > cols-1: 
            return False
        return True 
    
    def ismerged(self,x,y):
        '''
        输出格子的大小或者是类型
        以及该单元格的位置
        '''
        merged_cells = self.merged
        for a,b,c,d in merged_cells:
            if x >= a and x < b and y >= c and y < d:
                r_s = b-a
                c_s = d-c
                space = r_s * c_s
#                if space <= 4:
#                    m_type = 'small'
#                elif c_s - r_s >= 5:
#                    m_type = 'verylong'
#                else:
#                    m_type = 'huge'
                    
                if r_s == 1 and (c_s == 2 or c_s == 3):
                    m_type = 'shortlong'
                elif c_s == 1 and (r_s == 2 or r_s == 3):
                    m_type = 'thin'
                elif c_s - r_s >= 5:
                    m_type = 'verylong'                    
                elif space <= 12:
                    m_type = 'midsize'
                else:
                    m_type = 'huge'
                    
                ###判断位置
                if c_s == 1: #宽度为一的竖条状,特征分为highup & highdown
                    if x == a:
                        m_pos = 'highup'
                    elif x == b - 1:
                        m_pos = 'highdown'
                    else:
                        m_pos = 'highmid'
                elif r_s == 1:#长度为一的长条状,特征分为longleft & longright
                    if y == c:
                        m_pos = 'longleft'
                    elif y == d - 1:
                        m_pos = 'longright'
                    else:
                        m_pos = 'longmid'
                else:
                    if(x,y) == (a,c):
                        m_pos = 'topleft'
                    elif(x,y) == (b-1,c):
                        m_pos = 'botleft'
                    elif(x,y) == (a,d-1):
                        m_pos = 'topright'
                    elif(x,y) == (b-1,d-1):
                        m_pos = 'botright'
                    else:
                        m_pos = 'middle'

                return (True,self.sheet.cell_value(a,c),m_type,m_pos)
        return (False,'','','') 
    
    def cell_type(self,data):
        '''
        input:str, current cell data 
        output: [str,int] the data type of this cell : 正整数，负数，小数，纯字符，字符跟数字混合，
        含有特殊符号，单位字典，空值，合并导致的空值. 以及长度（不用于训练，用于提取特征）
        integer/dec/neg/string/mixed/specical/null/  
        int
        表外空值 这个特征在后面的函数中计算
        merged 提前判断
        '''

        def mixed(a):
            for i in a:
                try:
                    float(i)
                    return True 
                except :
                    pass
            return False 
        
        def puncuation(a):
            for i in a:
                if (str(i) in string.punctuation):
                    return True 
            return False 
        
        
        if data == '' or data == '\n':
            return 'null' 
        if isinstance(data,int): 
            return 'integer'
        elif isinstance(data,float): 
            return 'decimal'
        
        try:
            if float(data) < 0:
                return 'neg' #至此不可能有负数
            if data.count('.') == 1:
                    return 'decimal'
            else:
                return 'integer'#至此不再有数字
            
        except:#string/mixed/special 
            if puncuation(data):
                return 'special'
            elif mixed(data):
                return 'mixed'
            else:
                return 'string'

    def lens(self,x,y):
        '''
        input:str,str
        output:str
        '''
        x = str(x)
        y = str(y)
        if len(x) == len(y):
            return 'same'
        elif 0 < len(x) - len(y) < 3 :
            return 'more'
        elif -3 < len(x) - len(y) < 0 :
            return 'less'
        elif len(x) - len(y) >= 3:
            return 'muchmore'
        elif len(x) - len(y) <= -3:
            return 'muchless'
        else:
            print(x)
            print(y)
            print(len(x) - len(y))
            print('asdasd')

    
    def neighbor(self,location,row,col,rows,cols):
        ####确定要检索的位置
        '''
        f1: if valid
        f2: if merged 
        f3: cell data type 
        '''
        if location == 'up':
            r = row - 2
            c = col - 1
        elif location == 'down':
            r = row
            c = col - 1
        elif location == 'left':
            r = row - 1 
            c = col - 2
        elif location == 'right':
            r = row - 1
            c = col 
        elif location == 'up2':
            r = row - 3
            c = col - 1
        elif location == 'down2':
            r = row + 1
            c = col - 1
        elif location == 'left2':
            r = row - 1 
            c = col - 3
        elif location == 'right2':
            r = row - 1
            c = col + 1         
        elif location == 'upl':
            r = row - 2
            c = col - 2 
        elif location == 'upr':
            r = row - 2
            c = col 
        elif location == 'downl':
            r = row 
            c = col - 2
        elif location == 'downr':
            r = row
            c = col 
            
            
        if self.valid(r,c,rows,cols):
            data = self.sheet.cell_value(r,c) # 当前检索到的邻居的数据
            f1 = 'valid'
            center_data = self.sheet.cell_value(row-1,col-1) #中心的数据
            
                
            
            boolean, merged_data,merged_type,merged_pos = self.ismerged(r,c)
            if boolean: # 单元格是合并过的
                f2 = 'merged'
                f3 = self.cell_type(merged_data)
                f5 = merged_type
                f6 = merged_pos
                f4 = self.lens(center_data,merged_data)  # 对于合并过的单元格，是比较有数据的那个单元格的数据与中心数据
                f7 = self.selflens(merged_data)
            else:  #单元格没有合并过
                f2 = 'single'
                f3 = self.cell_type(data)
                f5 = 'singletype'
                f6 = 'singlepos'
                f4 = self.lens(center_data,data) #对于非合并的，操作自己与中心
                f7 = self.selflens(data)
            
        else:
            f1 = f2 = f3 = f4 = f5 = f6 = f7 = 'invalid'
        
        ls = location
        return [ls + f1, ls + f2, ls + f3 ,ls + f5, ls + f6 ,ls + f7]   #删除了f4
    
    '''
    def get_lens_features(self,f1,f2,location):
        
#        input: feature1 feature2
#        f1: current_cell 
#        f2: neighbor_cell
#        output: the dif of lens 
        
        if f1 in f2:
            return [location + self.lens(f1,f2)]
        else:
            return [location + 'diftype']
    '''
    
    def get_features (self, current_pos):
        '''
        给定想要提取特征的位置，以及他的邻居们，得到输出
        current_pos: (row,col)
        neighbors: locations ['up','down','left','right'.....]
        '''
        r , c = current_pos
        r = r+1
        c = c+1
        f = []
        #f1 = self.cell_type(self.sheet.cell_value(r-1,c-1)) #### 可以整的好看点
        
        boolean, merged_data,merged_type,merged_pos = self.ismerged(r-1,c-1)
        if boolean:
            f1 = self.cell_type(merged_data)
            f2 = 'merged'
            f3 = merged_type
            f4 = merged_pos
            f5 = self.selflens(merged_data)
        else:
            f1 = self.cell_type(self.sheet.cell_value(r-1,c-1))
            f2 = f3 = f4 = 'single'
            f5 = self.selflens(self.sheet.cell_value(r-1,c-1))
            
        
        
        f = f + [f1,f2,f3,f4,f5]
        neighbors = self.neighbors
        for i in neighbors:
            fff = self.neighbor(i,r,c,self.rows,self.cols)
            f = f + fff 
        return f
    
    def get_features_map (self): # 可以优化一波nparray
        '''
        提取整个sheet的所有位置的所有特征
        暂时不考虑空值的特征
        '''
        d = {}
        
        for i in range(self.rows):
            for j in range(self.cols):
                d[(i,j)] = self.get_features( (i,j) )

        return d
                
'''
a = Sheet(p)
x = a.get_features_map()        
'''
        
        
        
        
        
        
    