import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\nitin\Downloads\\Python_Unsupervised\\Python_Unsupervised\\DT , Bagging and Boosting  -Random Forest\\Py Code\\iris.csv")
data.head()
data['Species'].unique()
data.Species.value_counts()
colnames = list(data.columns)
predictors = colnames[:4]
target = colnames[4]

# Splitting data into training and testing data set
import numpy as np
from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])

preds = model.predict(test[predictors]) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 

pd.crosstab(test[target],preds) # getting the 2 way table to understand the correct and wrong predictions

# Accuracy 
np.mean(preds==test.Species) # 96.66



