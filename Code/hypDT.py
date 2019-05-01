
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import pandas as pd

#Code for Dataset - 1

dataset = pd.read_csv('C:/Users/Ezhilmani Aakaash/My Documents/LiClipse Workspace/AI_Mp_Two/ds1/ds1Train.csv',header=None)
valset = pd.read_csv('C:/Users/Ezhilmani Aakaash/My Documents/LiClipse Workspace/AI_Mp_Two/ds1/ds1Val.csv',header=None)

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
valX, valY = valset.iloc[:, :-1], valset.iloc[:, -1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

pred = clf.predict(valX)
accuracy = clf.score(valX, valY)
expec = valY

#predCSV=pd.DataFrame(pred,columns=None)
#predCSV.to_csv('ds1Val-dt.csv',header=False,encoding="utf-16")

print(float(accuracy))

print(metrics.confusion_matrix(expec, pred))
print(metrics.classification_report(expec, pred))

param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20,100,300],
              "max_depth": [None, 2, 5, 10,100,200],
              "min_samples_leaf": [1, 5, 10,50,100],
              "max_leaf_nodes": [None, 5, 10, 20,100],
              }

#param_grid = {'C':[0.001,0.1,1,10,100,1000],'gamma':[1,0.1,0.001,0.0001,1e-2,1e-4], 'kernel':['linear','rbf','sigmoid']}
grid = GridSearchCV(tree.DecisionTreeClassifier(),param_grid,refit = True, verbose=2,cv=4)
grid.fit(X,y)

print(grid.best_params_)
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]



