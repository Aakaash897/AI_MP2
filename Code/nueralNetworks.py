from sklearn import metrics
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import pandas as pd

#Code for Dataset - 1

dataset = pd.read_csv("ds1Train.csv",header=None)
valset = pd.read_csv("ds1Val.csv",header=None)
testset = pd.read_csv("ds1Test.csv",header=None)

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
valX, valY = valset.iloc[:, :-1], valset.iloc[:, -1]
testX = testset.iloc[:, :]

clf = MLPClassifier(hidden_layer_sizes=35,alpha=1.0)
clf = clf.fit(X,y)
joblib.dump(clf, 'ann_ds1.joblib')


pred = clf.predict(valX)
accuracy = clf.score(valX, valY)
expec = valY

predCSV=pd.DataFrame(pred,columns=None)
predCSV.to_csv('ds1Val-nt.csv',header=False,encoding="utf-16")

print(float(accuracy))

print(metrics.confusion_matrix(expec, pred))
print(metrics.classification_report(expec, pred))
print(metrics.matthews_corrcoef(expec, pred))




pred = clf.predict(testX)

predCSV=pd.DataFrame(pred,columns=None)
predCSV.to_csv('ds1Test-nt.csv',header=False,encoding="utf-16")




#Code for Dataset - 2

#dataset = pd.read_csv('C:/Users/Ezhilmani Aakaash/My Documents/LiClipse Workspace/AI_Mp_Two/ds2/ds2Train.csv',header=None)
#valset = pd.read_csv('C:/Users/Ezhilmani Aakaash/My Documents/LiClipse Workspace/AI_Mp_Two/ds2/ds2Val.csv',header=None)
#testset = pd.read_csv('C:/Users/Ezhilmani Aakaash/My Documents/LiClipse Workspace/AI_Mp_Two/ds2/ds2Test.csv',header=None)

dataset = pd.read_csv("ds2Train.csv",header=None)
valset = pd.read_csv("ds2Val.csv",header=None)
testset = pd.read_csv("ds2Test.csv",header=None)

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
valX, valY = valset.iloc[:, :-1], valset.iloc[:, -1]
testX = testset.iloc[:, :]

#criterion="entropy",splitter="best",min_samples_split=20,max_leaf_nodes=None,max_depth=50

#{'criterion': 'gini', 'max_depth': None, 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 20}
clf = MLPClassifier(hidden_layer_sizes=100,alpha=1.0)
clf = clf.fit(X,y)
joblib.dump(clf, 'ann_ds2.joblib')


pred = clf.predict(valX)
accuracy = clf.score(valX, valY)
expec = valY


predCSV=pd.DataFrame(pred,columns=None)
predCSV.to_csv('ds2Val-nt.csv',header=False,encoding="utf-16")

print(float(accuracy))
print(metrics.confusion_matrix(expec, pred))
print(metrics.classification_report(expec, pred))

pred = clf.predict(testX)

predCSV=pd.DataFrame(pred,columns=None)
predCSV.to_csv('ds2Test-nt.csv',header=False,encoding="utf-16")


#,'binarize':[None,0.5,1.0,10.0], 'fit_prior':['True','False']
param_grid = {'alpha':[0.001,0.5,1.0,10,100]}
#parameters = {'solver': ['adam'], 'max_iter': [500,1000,1500], 'alpha': 10.0 ** -np.arange(1, 3), 'hidden_layer_sizes':np.arange(5, 7), 'random_state':[0,1,2,3,4]}
#grid = GridSearchCV(MLPClassifier(),param_grid,verbose=2,n_jobs=3)
#grid.fit(X,y)

#print(grid.best_params_)
#pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
