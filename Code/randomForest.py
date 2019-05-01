
from sklearn import metrics
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import pandas as pd

#Code for Dataset - 1

dataset = pd.read_csv("ds1Train.csv",header=None)
valset = pd.read_csv("ds1Val.csv",header=None)
testset = pd.read_csv("ds1Test.csv",header=None)

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
valX, valY = valset.iloc[:, :-1], valset.iloc[:, -1]
testX = testset.iloc[:, :]

clf = RandomForestClassifier(criterion="entropy",min_samples_split=2,max_leaf_nodes=None,max_depth=None,min_samples_leaf= 1)
clf = clf.fit(X,y)
joblib.dump(clf, 'rf_ds1.joblib')


pred = clf.predict(valX)
accuracy = clf.score(valX, valY)
expec = valY

predCSV=pd.DataFrame(pred,columns=None)
predCSV.to_csv('ds1Val-rf.csv',header=False,encoding="utf-16")

print(float(accuracy))

print(metrics.confusion_matrix(expec, pred))
print(metrics.classification_report(expec, pred))
print(metrics.matthews_corrcoef(expec, pred))




pred = clf.predict(testX)

predCSV=pd.DataFrame(pred,columns=None)
predCSV.to_csv('ds1Test-rf.csv',header=False,encoding="utf-16")




#Code for Dataset - 2

dataset = pd.read_csv("ds2Train.csv",header=None)
valset = pd.read_csv("ds2Val.csv",header=None)
testset = pd.read_csv("ds2Test.csv",header=None)

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
valX, valY = valset.iloc[:, :-1], valset.iloc[:, -1]
testX = testset.iloc[:, :]

#criterion="entropy",splitter="best",min_samples_split=20,max_leaf_nodes=None,max_depth=50

#{'criterion': 'gini', 'max_depth': None, 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 20}
clf = RandomForestClassifier(criterion="gini",min_samples_split=20,max_leaf_nodes=None,max_depth=None,min_samples_leaf= 1)
clf = clf.fit(X,y)
joblib.dump(clf, 'rf_ds2.joblib')


pred = clf.predict(valX)
accuracy = clf.score(valX, valY)
expec = valY


predCSV=pd.DataFrame(pred,columns=None)
predCSV.to_csv('ds2Val-rf.csv',header=False,encoding="utf-16")

print(float(accuracy))
print(metrics.confusion_matrix(expec, pred))
print(metrics.classification_report(expec, pred))

pred = clf.predict(testX)

predCSV=pd.DataFrame(pred,columns=None)
predCSV.to_csv('ds2Test-rf.csv',header=False,encoding="utf-16")
