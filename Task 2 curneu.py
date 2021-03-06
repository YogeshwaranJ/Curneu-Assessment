# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 18:31:20 2021

@author: yoges
"""

#importing all the needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from scipy.stats import mode 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 


#reading the data using pandas
diabetes_dat = pd.read_csv('C:/Users/yoges/Documents/SD03Q03/SD03Q03/Diabetes Database.csv')

