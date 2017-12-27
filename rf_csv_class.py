# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 17:26:25 2017

@author: cisdi
"""

'''
训练rf模型
用rf模型分类csv文件
'''

import xlrd

import string
import pandas as pd 
import csv
import os
import random
import numpy as np

'''
定义了sheet类
包含了用xlrd解析出的各种信息以及提取特征用的函数
'''



class table_csv:
    def __init__(self, file_path, merged_path):
        
#        self.table_data  = np.loadtxt(file_path, dtype=np.str, delimiter=",")
        df=pd.read_csv(file_path, header= None, encoding="utf-8")  #读取csv文件

        self.table_data  = np.array(df) 
        self.rows = self.table_data.shape[0]
        self.cols = self.table_data.shape[1]
        
        self.neighbors = ['up','down','left','right','up2','down2']
        self.num_features = 41
        self.features = []
        
        
        df2=pd.read_csv(merged_path, header= None, encoding="utf-8")
        m = np.array(df2) 
        rr = m.shape[0]
        self.merged = set()
#        for i in range(self.rows):
        for i in range(rr):   
            a,b,c,d = m[i]   ######a,b,c,d都为闭区间且从零开始
            self.merged.add((a,c+1,b,d+1)) # 转化为x1,x2,y1,y2 且x2,y2为闭区间
        

#        self.neighbors = ['up','down','left','right','up2','down2','left2','right2','upl','upr','downl','downr']

#        self.neighbors = ['up','down','left','right',]
    

    def nan(self,data):
        try:
            return np.isnan(data)
        except:
            return False
        
        
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
        if self.nan(data):
            return 'dataveryshort'
        
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
    

    
    def valid(self,x,y): # 可以优化
        if x < 0 or y < 0 or x > self.rows-1 or y > self.cols-1: 
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
                
                if c_s == self.cols or c_s == self.cols -1 :
                    m_type = 'wholeline'
#                
                elif r_s == 1 and (c_s == 2 or c_s == 3):
                    m_type = 'shortlong'
                elif c_s == 1 and (r_s == 2 or r_s == 3):
                    m_type = 'thin'
                elif c_s == 1:
                    m_type = 'thinhigh'
                elif c_s - r_s >= 5:
                    m_type = 'verylong'                    
                elif space <= 12:
                    m_type = 'midsize'
                else:
                    m_type = 'huge'
#                m_type = str(c_s)
                    
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

                return (True,self.table_data[a,c],m_type,m_pos,(a,b,c,d))
        return (False,'','','','') 
    
    
    
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
        
        
        if self.nan(data) or data == '\n':
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
    
    def find_neighbor(self,x,y,x1,y1):
        '''
        (2tuple,2tuple,4tuple) --> 2tuple
        input: center postion, current postion, the merged cell info of center postioin, output the true postion of neighbor
        '''
        boolean, merged_data,merged_type,merged_pos,merged_cell = self.ismerged(x,y)
        
        
        xx = x1 - x
        yy = y1 - y # 获取移动方向
        if not boolean:
            return (x1,y1)
        else:            
            a,b,c,d = merged_cell
            while 1:
                if x1 >= a and x1 < b and y1 >= c and y1 < d: # 在同一个cell里面
                    x1 = x1 + xx #朝计算出的方向移动
                    y1 = y1 + yy
                else:
                    return (x1,y1)
    
    def neighbor_features(self,row,col,rows,cols):
        r = row - 1
        c = col - 1 ##center pos
        up_pos = self.find_neighbor((r,c),(r-1,c))
        return ''
        
        
    
    def neighbor(self,location,row,col,rows,cols):
        ####确定要检索的位置
        '''
        f1: if valid
        f2: if merged 
        f3: cell data type 
        f4: deleted
        f5: type of merged cell
        f6: pos in the merged cell
        f7: lens of the data
        f8: data(for test)
        '''
        row = row - 1
        col = col - 1
        if location == 'up':
            (r,c) = self.find_neighbor(row,col,row-1,col)
        elif location == 'down':
            (r,c) = self.find_neighbor(row,col,row+1,col)
        elif location == 'left':
            (r,c) = self.find_neighbor(row,col,row,col-1)
        elif location == 'right':
            (r,c) = self.find_neighbor(row,col,row,col+1)
        elif location == 'up2':
            (r,c) = self.find_neighbor(row,col,row-1,col)
            (r,c) = self.find_neighbor(r,c,r-1,c)
        elif location == 'down2':
            (r,c) = self.find_neighbor(row,col,row+1,col)
            (r,c) = self.find_neighbor(r,c,r+1,c)
        elif location == 'left2':
            (r,c) = self.find_neighbor(row,col,row,col-1)
            (r,c) = self.find_neighbor(r,c,r,c-1)
        elif location == 'right2':
            (r,c) = self.find_neighbor(row,col,row,col+1)
            (r,c) = self.find_neighbor(r,c,r,c+1)       
#        elif location == 'upl':
#            r = row - 2
#            c = col - 2 
#        elif location == 'upr':
#            r = row - 2
#            c = col 
#        elif location == 'downl':
#            r = row 
#            c = col - 2
#        elif location == 'downr':
#            r = row
#            c = col 
        elif location == 'up3':
            (r,c) = self.find_neighbor(row,col,row-1,col)
            (r,c) = self.find_neighbor(r,c,r-1,c)
            (r,c) = self.find_neighbor(r,c,r-1,c)
        elif location == 'down3':
            (r,c) = self.find_neighbor(row,col,row+1,col)
            (r,c) = self.find_neighbor(r,c,r+1,c)
            (r,c) = self.find_neighbor(r,c,r+1,c)
             
            
            
        if self.valid(r,c):
            data = self.table_data[r,c] # 当前检索到的邻居的数据
            f1 = 'valid'
            center_data = self.table_data[row-1,col-1] #中心的数据

            boolean, merged_data,merged_type,merged_pos,cell_pos = self.ismerged(r,c)
            #merged_data 合并单元格的数据
            if boolean: # 单元格是合并过的
                f2 = 'merged'
                f3 = self.cell_type(merged_data)
                f5 = merged_type
                f6 = merged_pos
#                f4 = self.lens(center_data,merged_data)  # 对于合并过的单元格，是比较有数据的那个单元格的数据与中心数据
                f7 = self.selflens(merged_data)
                f8 = merged_data
            else:  #单元格没有合并过
                f2 = 'single'
                f3 = self.cell_type(data)
                f5 = 'singletype'
                f6 = 'singlepos'
#                f4 = self.lens(center_data,data) #对于非合并的，操作自己与中心
                f7 = self.selflens(data)
                f8 = data

            
        else:
            f1 = f2 = f3 = f4 = f5 = f6 = f7 = f8 = 'invalid'

        
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
        
        boolean, merged_data,merged_type,merged_pos,cell_pos = self.ismerged(r-1,c-1)
        if boolean:
            f1 = self.cell_type(merged_data)
            f2 = 'merged'
            f3 = merged_type
            f4 = merged_pos
            f5 = self.selflens(merged_data)
        else:
            f1 = self.cell_type(self.table_data[r-1,c-1])
            f2 = f3 = f4 = 'single'
            f5 = self.selflens(self.table_data[r-1,c-1])
        f = f + [f1,f2,f3,f4,f5] 
        
        
        
        neighbors = self.neighbors ###计算neighbors的特征
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
    def get_features_map_dataframe (self):
        header = ['pos']
        f_count = 1
        for i in range(self.num_features):
            header = header + ['f'+str(f_count)]
            f_count +=1
            
        #df = pd.DataFrame(columns = header)
        
        a = []
        for i in range(self.rows):
            for j in range(self.cols):
                a1 = [(i,j)]
                a2 = self.get_features( (i,j) )
                if a2[0] != 'null':
                    a.append ( a1+ a2)
                
        #df.append(a)
        df = pd.DataFrame(a, columns = header)
        return df

'''
xx =   table_csv ('test1.csv' , 'test2.csv')
        
a = xx.get_features_map()
b = xx.get_features_map_dataframe()
'''
def rule1(d,merged):
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
        
    