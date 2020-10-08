# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 18:45:22 2020

@author: nitin
"""

##50 STARTUP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('C:\\Users\\nitin\\Desktop\\Assignments\\Multiple Linear Regression\\50_Startups.csv')
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]

#converting categorical data into separate columns (one hot encoding)
states = pd.get_dummies(x['State'])

#dropping the original states column
x = x.drop('State', axis = 1)

#concatenating dummy variable and the original data 
x = pd.concat([x,states], axis = 1)

#splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#fitting multiple linear regression to the training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the accuracy
from sklearn.metrics import r2_score
y_pred = regressor.predict(x_test)
accuracy = r2_score(y_test, y_pred)
print(accuracy)
#with accuracy of 0.93, we have built a very good model

##PREDICTING THE PRICE OF THE COMPUTER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
comp_dataset = pd.read_csv('C:\\Users\\nitin\\Desktop\\Assignments\\Multiple Linear Regression\\Computer_Data.csv')
cols = [0,10]
comp_df = comp_dataset.drop(comp_dataset.columns[cols], axis = 1)

#checking null values
comp_dataset.isnull()
#there are no null values, so we'll proceed with the data categorisation into X and Y variables.
x = comp_dataset.iloc[:,[1,2,3,4,5,6,7,8]]
y = comp_dataset.iloc[:,0]

#converting categorical data into dummy variables.
dummy = pd.get_dummies(data = x, columns =['cd','multi','premium'])
x = dummy

#splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#fitting multiple linear regression to the training data
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

#predicting the accuracy
from sklearn.metrics import r2_score
y_pred = reg.predict(x_test)
acc = r2_score(y_test, y_pred)
print(acc)
#with 53% accuracy the model is not very robust.


