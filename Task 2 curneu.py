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

#description of the dataset
print(diabetes_dat.describe())

#finding the number of missing values
db_dat_copy = diabetes_dat.copy(deep = True)
db_dat_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = db_dat_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(db_dat_copy.isnull().sum())



#visualization of the outlier in the data
import seaborn as sns
sns.boxplot(x=diabetes_dat["Pregnancies"])

sns.boxplot(x=diabetes_dat["Glucose"])

sns.boxplot(x=diabetes_dat['BloodPressure'])

sns.boxplot(x=diabetes_dat['SkinThickness'])

sns.boxplot(x=diabetes_dat['Insulin'])

sns.boxplot(x=diabetes_dat['BMI'])

sns.boxplot(x=diabetes_dat['DiabetesPedigreeFunction'])

sns.boxplot(x=diabetes_dat['Age'])



#removing the outliers
parameters = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
for i in parameters:#for loop to iterate over all the parameters to remove outliers
    Q1 = diabetes_dat[i].quantile(0.25)
    Q3 = diabetes_dat[i].quantile(0.75)
    IQR = Q3 - Q1
    diabetes_dat[i] = diabetes_dat[i][~((diabetes_dat[i] < (Q1 - 1.5 * IQR)) |(diabetes_dat[i] > (Q3 + 1.5 * IQR)))]


#gathering the info of the data after removing outliers
print(diabetes_dat.info(verbose=True))



#droping all the nan values
diabetes_dat.dropna(subset =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'] , inplace=True)
diabetes_dat.info(verbose=True)



    