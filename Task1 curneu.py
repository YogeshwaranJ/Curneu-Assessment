# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:00:01 2021

@author: yoges
"""
#importing all the needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#reading the data using pandas
dat = pd.read_excel('C:/Users/yoges/Documents/SD03Q03/SD03Q03/fruits.xlsx')

#viewing the first 5 data
print(dat.head(5))

#description of the data
print(dat.describe())

#plotting various combinations of parameter
plt.scatter(dat['mass'], dat['width'])
plt.xlabel('Mass', )
plt.ylabel('width')
plt.title('Mass vs width')


plt.scatter(dat['mass'], dat['height'])
plt.xlabel('Mass')
plt.ylabel('height')
plt.title('Mass vs height')

plt.scatter(dat['mass'], dat['color_score'])
plt.xlabel('Mass')
plt.ylabel('Color_score')
plt.title('Mass vs Color_score')

#finding the correlation amoung the parameters
print(dat.iloc[:,1:].corr())

features = ['width','color_score']
X = dat[features]
y = dat.iloc[:,0]
