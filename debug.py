# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 08:54:28 2017

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
from rf_csv_class import *

     
x = Sheet('C:/Users/cisdi/Desktop/test_for_max/27.xls')
#
#
##x.sheet.cell_value(0,24)
##x.get_features((0,24))
#asd = x.get_features_map()
#asdd = x.get_features_map_dataframe()

aaa  = x.get_features_map_dataframe()

xx = table_csv ('test1.csv' , 'test2.csv')

asd = xx.get_features_map()

asdf = xx.get_features_map_dataframe()


a = xx.get_features((8,11))
b = x.get_features((8,11))