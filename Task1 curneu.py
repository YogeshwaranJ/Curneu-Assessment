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


plt.scatter(dat['width'], dat['color_score'])
plt.xlabel('Width')
plt.ylabel('Color_score')
plt.title('Width vs Color_score')

#finding the correlation amoung the parameters
print(dat.iloc[:,1:].corr())

features = ['width','color_score']
X = dat[features]
y = dat.iloc[:,0]


#splitting of dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)

#standardize the training and test dataset

from sklearn.preprocessing import StandardScaler
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
        
        # Storing distances in a dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'], 
                                index=y_train.index)
        
        # Sort distances and considering the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])

        # Get most common label of all the nearest neighbors
        prediction = counter.most_common()[0][0]
        
        # Append prediction to output list
        y_hat_test.append(prediction)
        
    return y_hat_test

#finding the best suitable k value
error = []
for i in range(1, 40):
    pred_i = knn_predict(X_train, X_test, y_train, y_test, k=i)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# predicting on test dataset
y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k=3)
print(y_test)
print(y_hat_test)
    
#calculating accuracy of the model built from scratch
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

print('Accuracy:', accuracy_score(y_test,y_hat_test))
print('r2 score:', r2_score(y_test,y_hat_test))


