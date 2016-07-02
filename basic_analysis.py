# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:02:39 2016

@author: joostbloom
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

'''
LOADING DATA
'''
from load_data import load_data
# Settings for loading data
maxRows = 20000 #<=0 to read all data
random_state = 99
reloadData = False

start = time.time()
(itemdata, path) = load_data(forceReload=reloadData,maxRows=maxRows, rs=random_state)
print("Loading data ({} rows) took {:.2f}s".format(data.shape[0], time.time()-start))


'''
FEATURE ENGINEERING
'''
from feature_engineering import load_features
recalculateFeatures = (True or reloadData)
start = time.time()
data = load_features(data, itemdata, forceReload = recalculateFeatures)
print("Loading features took {:.2f}s".format(time.time()-start))

selected_features = ['d_price','d_title','d_len_title','d_description','d_len_description','d_images','d_len_images']
#selected_features = ['d_price','d_title','d_description']

'''
SPLITTING DATASETS
'''
from sklearn.cross_validation import train_test_split
   
y = data.isDuplicate
X = data.drop(list(set(data.columns) - set(selected_features)),axis=1)
#
X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.2, random_state=random_state)


'''
TRAIN DECISION TREE
'''
from decisiontree import predict_using_decision_tree, report_result
forceUpdateClf = (False or recalculateFeatures)

clf, y_predict = predict_using_decision_tree(X_train, y_train, X_val, forceUpdateClf)

report_result(y_val, y_predict)


'''
KNN Very low rate of duplicates, high rate of no duplicates
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=500, weights='uniform')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_val)
report_result(y_pred,y_val)

LINEAR MODEL same issue
from sklearn import linear_model
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_val)
report_result(y_pred,y_val)
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, grid_search
rf = RandomForestClassifier()
param_grid = {"max_depth": [3, 4, 5],
              "max_features": [3, 5],
              "min_samples_split": [20, 40],
              "min_samples_leaf": [40, 80],
              "bootstrap": [True],
              "criterion": ["gini"]}
cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=100,test_size=0.2, random_state=123)
clf = grid_search.GridSearchCV(rf, param_grid, cv=cv, n_jobs=4, scoring='roc_auc')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_val)
report_result(y_pred,y_val)
'''
RUN ON TEST DATA
'''
from load_data import load_data_test
from decisiontree import predict_result
(testdata,testitemdata) = load_data_test()

testdata = load_features(testdata, traindata=False, forceReload = True)

X_test = testdata.drop(list(set(testdata.columns) - set(selected_features)),axis=1)
X_test.d_price = X_test.d_price.fillna(0)

y_test = predict_result(clf, X_test)

y_prob = clf.predict_proba(X_test)


'''
FINISH AND CREATE SUBMIT FILE
'''
print("Creating submssion file")
from save_data import create_submit_file_kaggle_avito
create_submit_file_kaggle_avito(y_prob)


#from sklearn.learning_curve import learning_curve
#from decisiontree import train_decision_tree
#from sklearn import cross_validation
#from sklearn import tree

#clf = train_decision_tree(X_train, y_train,forceUpdateClf)
#start = time.time()
#cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=100,test_size=0.2, random_state=random_state)
#plot_learning_curve(tree.DecisionTreeClassifier(), "Testje",X, y, cv=cv)
#plot_learning_curve(clf, "Testje",X, y, cv=cv, n_jobs=4)
#print("Learning curve took {:.2f}s".format(time.time()-start))
#plotDecisionBoundary(X,y,'d_price',clf2)

def plotDecisionBoundary(X,y,c_all,clf):
    #x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    NCol = 3
    NRow = X.shape[1] / NCol
    ncol = 0
    nrow = 0
    
    if NRow == 1: NRow = 2
    
    f, axarr = plt.subplots(NRow, NCol, figsize=(10, 8))
    
    
    for c in X.columns:
        
        
        X1 = X[c_all]
        X2 = X[c]
        print('Feature {} assumed median {:.2f}'.format(c,X[c].median()))
        
        if c==c_all:
            continue
        
            
        #prop_test = 0.2
        #X_train, X_test, y_train, y_test = train_test_split(X[ [c, c_all]], y, test_size=prop_test, random_state=123)

        #clf.fit(X_train, y_train)
        
        res = 0.1
    
        x_min, x_max = changeby(min(X1),-2*res), changeby(max(X1),2*res)
        y_min, y_max = changeby(min(X2),-2*res), changeby(max(X2),2*res)
    
    
        xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                             np.arange(y_min, y_max, res))              
    
    #
        Xplot = pd.DataFrame(data=None, columns=X.columns)
        Xplot[c_all] = xx.ravel()
        for cc in Xplot.columns:
            Xplot[cc] = X[cc].median()
        
        Xplot[c_all] = xx.ravel()
        Xplot[c] = yy.ravel()
            
    
        Z = clf.predict(Xplot)
        #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        #print ncol, nrow, NCol
    
        axarr[nrow, ncol].contourf(xx, yy, Z, alpha=0.4)
        axarr[nrow, ncol].scatter(X1, X2, c=y, alpha=0.5)
        axarr[nrow, ncol].set_title(clf.__class__.__name__)
        axarr[nrow, ncol].set_ylim([min(X2), max(X2)]) 
        axarr[nrow, ncol].set_xlim([min(X1), max(X1)]) 
        axarr[nrow, ncol].set_xlabel(c_all)
        axarr[nrow, ncol].set_ylabel(c)    
        
        
        ncol += 1
        if ncol>=NCol:
            ncol = 0
            nrow += 1
       
    plt.title(clf.__class__.__name__) 
    plt.show()
    
def changeby(value,factor):
    return value + factor*abs(value) if value!=0 else factor
    






