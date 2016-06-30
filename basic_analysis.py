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
maxRows = 10000 #None to read all data
random_state = 99
reloadData = False

start = time.time()
(data, itemdata) = load_data(forceReload=reloadData,maxRows=maxRows, rs=random_state)
print("Loading data ({} rows) took {:.2f}s".format(data.shape[0], time.time()-start))


'''
FEATURE ENGINEERING
'''
from feature_engineering import load_features
recalculateFeatures = (False or reloadData)
start = time.time()
data = load_features(data, itemdata, recalculateFeatures)
print("Loading features took {:.2f}s".format(time.time()-start))

selected_features = ['d_price','d_title','d_len_title','d_description','d_len_description','d_images','d_len_images']
#selected_features = ['d_price','d_title','d_description']

'''
SPLITTING DATASETS
'''
from sklearn.cross_validation import train_test_split
   
y = data.isDuplicate
X = data.drop(list(set(data.columns) - set(selected_features)),axis=1)
cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=100,
                                   test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=random_state)

'''
TESTS USING DECISION TREE
'''
from decisiontree import predict_using_decision_tree, report_result
forceUpdateClf = (False or recalculateFeatures)

y_predict = predict_using_decision_tree(X_train, y_train, X_test, forceUpdateClf)

report_result(y_test, y_predict)

from sklearn.learning_curve import learning_curve
from decisiontree import train_decision_tree

clf = train_decision_tree(X_train, y_train)
learning_curve(clf, X, y)
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
    






