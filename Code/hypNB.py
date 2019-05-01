
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
import pandas as pd

#Code for Dataset - 1

dataset = pd.read_csv('C:/Users/Ezhilmani Aakaash/My Documents/LiClipse Workspace/AI_Mp_Two/ds2/ds2Train.csv',header=None)
valset = pd.read_csv('C:/Users/Ezhilmani Aakaash/My Documents/LiClipse Workspace/AI_Mp_Two/ds2/ds2Val.csv',header=None)

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
valX, valY = valset.iloc[:, :-1], valset.iloc[:, -1]

#C=10, gamma=0.001, kernel='rbf'
clf = BernoulliNB(alpha=0.001, binarize=None, fit_prior='True')
clf = clf.fit(X,y)


pred = clf.predict(valX)
accuracy = clf.score(valX, valY)
expec = valY

print(float(accuracy))

predCSV=pd.DataFrame(pred,columns=None)
predCSV.to_csv('ds1Val-3.csv',header=False,encoding="utf-16")

print(metrics.confusion_matrix(expec, pred))
print(metrics.classification_report(expec, pred))

clf = BernoulliNB()

#alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
param_grid = {'alpha':[0.001,0.5,1.0,10,100],'binarize':[None,0.5,1.0,10.0], 'fit_prior':['True','False']}

#grid = GridSearchCV(BernoulliNB(),param_grid,refit = True, verbose=2)
#grid.fit(X,y)

#print(grid.best_params_)
#pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]



