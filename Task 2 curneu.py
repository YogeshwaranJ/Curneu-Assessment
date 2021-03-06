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

#selecting the features and seperating features and target variables
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_dat[features]
y = diabetes_dat.iloc[:,8]


#splitting of dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)


#standardizing the training and testing data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



#function to calculate euclidean distance
def euclid_dist(X1,X2):
    dist = np.sum((X1 - X2)**2)
    return np.sqrt(dist)

#function of knn predict
def knn_predict(X_train, X_test, y_train, y_test, k):
    
    # Counter to help with label voting
    from collections import Counter
    
    # Make predictions on the test data
    # Need output of 1 prediction per test data point
    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = euclid_dist(test_point, train_point)
            distances.append(distance)
        
        # Store distances in a dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'], 
                                index=y_train.index)
        
        # Sort distances, and only consider the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])

        # Get most common label of all the nearest neighbors
        prediction = counter.most_common()[0][0]
        
        # Append prediction to output list
        y_hat_test.append(prediction)
        
    return y_hat_test


# Make predictions on test dataset
y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k=20)

    