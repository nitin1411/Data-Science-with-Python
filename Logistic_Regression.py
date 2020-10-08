# Logistic Regression Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
dataframe = read_csv(filename, names=name)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
mod=model.fit(X,Y)
prediction=mod.predict(X)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
