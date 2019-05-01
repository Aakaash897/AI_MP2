
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd

#Code for Dataset - 1

dataset = pd.read_csv('C:/Users/Ezhilmani Aakaash/My Documents/LiClipse Workspace/AI_Mp_Two/ds2/ds2Train.csv',header=None)
valset = pd.read_csv('C:/Users/Ezhilmani Aakaash/My Documents/LiClipse Workspace/AI_Mp_Two/ds2/ds2Val.csv',header=None)

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
valX, valY = valset.iloc[:, :-1], valset.iloc[:, -1]

#C=10, gamma=0.001, kernel='rbf'
#clf = SVC()
#clf = clf.fit(X,y)


#pred = clf.predict(valX)
#accuracy = clf.score(valX, valY)
#expec = valY

#print(float(accuracy))

#predCSV=pd.DataFrame(pred,columns=None)
#predCSV.to_csv('ds1Val-3.csv',header=False,encoding="utf-16")

#print(metrics.confusion_matrix(expec, pred))
#print(metrics.classification_report(expec, pred))

#clf = SVC()

param_grid = {'C':[0.001,0.1,1,10,100,1000],'gamma':[1,0.1,0.001,0.0001,1e-2,1e-4], 'kernel':['linear','rbf','sigmoid']}
grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)
grid.fit(X,y)

print(grid.best_params_)
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]



