# SVM Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
dataframe = read_csv(filename, names=name)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = SVC()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


#classification of forest fires using SVM
data = read_csv("C:\\Users\\nitin\\Desktop\\Assignments\\SVM\\forestfires.csv")
arr = data.values
x = arr[:,2:30]
y = arr[:,30]
kfold = KFold(n_splits = 10, random_state = 7 )
mod = SVC()
result = cross_val_score(mod, x,y, cv = kfold)
print(result)


#classification of salary data using SVM
from pandas import get_dummies
sal_data = read_csv("C:\\Users\\nitin\\Desktop\\Assignments\\SVM\\SalaryData_Train(1).csv")
train_sd = get_dummies(sal_data) #creating dummy variables on train data
train_sd.head()

test_sal_data = read_csv("C:\\Users\\nitin\Desktop\\Assignments\\SVM\\SalaryData_Test(1).csv")
test_sd = get_dummies(test_sal_data)
arr_train = train_sd.values
arr_test = test_sd.values
x_train = arr_train[:,0:103]
y_train = arr_train[:,103:105]
kf = KFold(n_splits = 10, random_state = 7)
mod1 = SVC()
result1 = cross_val_score(mod1, x_train, y_train, cv = kf)
print(result1)

#model validation on test data
x_test = arr_test[:,0:103]
y_test = arr_test[:,103:105]
result2 = cross_val_score(mod1, x_test, y_test, cv = kf)
print(result2)
