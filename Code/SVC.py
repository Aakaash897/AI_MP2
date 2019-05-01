
from sklearn import metrics
from sklearn.svm import SVC
from scipy.stats import randint
from sklearn.externals import joblib
import pandas as pd

#Code for Dataset - 1

dataset = pd.read_csv("ds1Train.csv",header=None)
valset = pd.read_csv("ds1Val.csv",header=None)
testset = pd.read_csv("ds1Test.csv",header=None)

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
valX, valY = valset.iloc[:, :-1], valset.iloc[:, -1]
testX = testset.iloc[:, :]

clf = SVC(C=10, gamma=0.001, kernel='rbf')
clf = clf.fit(X,y)
joblib.dump(clf, 'svm_ds1.joblib')

pred = clf.predict(valX)
accuracy = clf.score(valX, valY)
expec = valY

print(float(accuracy))

predCSV=pd.DataFrame(pred,columns=None)
predCSV.to_csv('ds1Val-3.csv',header=False,encoding="utf-16")

print(metrics.confusion_matrix(expec, pred))
print(metrics.classification_report(expec, pred))

pred = clf.predict(testX)

predCSV=pd.DataFrame(pred,columns=None)
predCSV.to_csv('ds1Test-3.csv',header=False,encoding="utf-16")





#Code for Dataset - 2

dataset = pd.read_csv("ds2Train.csv",header=None)
valset = pd.read_csv("ds2Val.csv",header=None)
testset = pd.read_csv("ds2Test.csv",header=None)


X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
valX, valY = valset.iloc[:, :-1], valset.iloc[:, -1]
testX = testset.iloc[:, :]

#{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
clf = SVC(C=10, gamma=0.01, kernel='rbf')
clf = clf.fit(X,y)
joblib.dump(clf, 'svm_ds2.joblib')

pred = clf.predict(valX)
accuracy = clf.score(valX, valY)
expec = valY

print(float(accuracy))

predCSV=pd.DataFrame(pred,columns=None)
predCSV.to_csv('ds2Val-3.csv',header=False,encoding="utf-16")

print(metrics.confusion_matrix(expec, pred))
print(metrics.classification_report(expec, pred))

pred = clf.predict(testX)

predCSV=pd.DataFrame(pred,columns=None)
predCSV.to_csv('ds2Test-3.csv',header=False,encoding="utf-16")
