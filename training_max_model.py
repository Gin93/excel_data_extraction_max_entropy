# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:39:38 2017

@author: cisdi
"""

# encoding=utf-8
# @Author: WenDesi
# @Date:   05-11-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   WenDesi
# @Last modified time: 09-11-16


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

class MaxEnt(object):
    
    def __init__(self):
    
        self.max_iteration = 100
        #引用外部w
        

    def init_params(self, X, Y):
        self.X_ = X
        self.Y_ = set()

        self.cal_Pxy_Px(X, Y)

        self.N = len(X)                 # 训练集大小
        self.n = len(self.Pxy)          # 书中(x,y)对数
        self.M = 10000.0                # 书91页那个M，但实际操作中并没有用那个值
        # 可认为是学习速率

        self.build_dict()
        self.cal_EPxy()

    def build_dict(self):
        self.id2xy = {}
        self.xy2id = {}

        for i, (x, y) in enumerate(self.Pxy):
            self.id2xy[i] = (x, y)
            self.xy2id[(x, y)] = i

    def cal_Pxy_Px(self, X, Y):
        self.Pxy = defaultdict(int)
        self.Px = defaultdict(int)

        for i in range(len(X)):
            x_, y = X[i], Y[i]
            self.Y_.add(y)

            for x in x_:
                self.Pxy[(x, y)] += 1
                self.Px[x] += 1

    def cal_EPxy(self):
        '''
        计算书中82页最下面那个期望
        '''
        self.EPxy = defaultdict(float)
        for id in range(self.n):
            (x, y) = self.id2xy[id]
            self.EPxy[id] = float(self.Pxy[(x, y)]) / float(self.N)

    def cal_pyx(self, X, y):
        result = 0.0
        for x in X:
            if self.fxy(x, y):
                id = self.xy2id[(x, y)]
                result += self.w[id]
        return (math.exp(result), y)

    def cal_probality(self, X):
        '''
        计算书85页公式6.22
        '''
        Pyxs = [(self.cal_pyx(X, y)) for y in self.Y_]
        Z = sum([prob for prob, y in Pyxs])
        return [(prob / Z, y) for prob, y in Pyxs]

    def cal_EPx(self):
        '''
        计算书83页最上面那个期望
        '''
        self.EPx = [0.0 for i in range(self.n)]

        for i, X in enumerate(self.X_):
            Pyxs = self.cal_probality(X)

            for x in X:
                for Pyx, y in Pyxs:
                    if self.fxy(x, y):
                        id = self.xy2id[(x, y)]

                        self.EPx[id] += Pyx * (1.0 / self.N)

    def fxy(self, x, y):
        return (x, y) in self.xy2id

    def train(self, X, Y):
        self.init_params(X, Y)
        self.w = [0.0 for i in range(self.n)]

        
        for times in range(self.max_iteration):
            print ('iterater times %d' % times)
            sigmas = []
            self.cal_EPx()

            for i in range(self.n):
                sigma = 1 / self.M * math.log(self.EPxy[i] / self.EPx[i])
                sigmas.append(sigma)

            # if len(filter(lambda x: abs(x) >= 0.01, sigmas)) == 0:
            #     break

            self.w = [self.w[i] + sigmas[i] for i in range(self.n)]

        
    def get_model(self,path,X,Y):
        '''
        读取已经训练好的模型
        '''
        reloaded_list = np.loadtxt(path)
        
        
        self.init_params(X, Y)
        self.w = reloaded_list.tolist()
        
        
        
    def predict(self, testset):
        results = []
        for test in testset:
            result = self.cal_probality(test)
            results.append(max(result, key=lambda x: x[0])[1])
        return results
    
    def predict1(self, one):

        result = self.cal_probality(one)
        a = (max(result, key=lambda x: x[0])[1])
        return a
    
    

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




if __name__ == "__main__":

    print ('Start read data')

    time_1 = time.time()
    raw_data = pd.read_csv('features.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.05, random_state=23323)

    train_features = rebuild_features(train_features)
    test_features = rebuild_features(test_features)

    time_2 = time.time()
    print ('read data cost ', time_2 - time_1, ' second', '\n')

    print ('Start training')
    met = MaxEnt()
    met.train(train_features, train_labels)

    time_3 = time.time()
    print ('training cost ', time_3 - time_2, ' second', '\n')
    
    #####训练完成 保存训练结果#######################
    numpy_list = np.asarray(met.w)
    model_path = "./training_results.txt"
    np.savetxt(model_path, numpy_list)
    
    

    print ('Start predicting')
    test_predict = met.predict(test_features)
    time_4 = time.time()
    print ('predicting cost ', time_4 - time_3, ' second', '\n')
    
    
    #统计结果
    correct = 0
    wrong = 0
    for i in range(len(test_labels)):
        if test_labels[i] == test_predict[i]:
            correct += 1
        else:
            wrong += 1 
    

    