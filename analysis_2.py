# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:43:10 2016

@author: joostbloom
"""

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
from load_data import load_train_data
# Settings for loading data
maxRows = 0 #<=0 to read all data
random_state = 99
reloadData = True

start = time.time()
(data, path) = load_train_data(forceReload=reloadData,maxRows=maxRows, rs=random_state)
print("Loading data ({} rows) took {:.2f}s".format(data.shape[0], time.time()-start))

'''
FEATURE ENGINEERING
'''
from feature_engineering import load_features
recalculateFeatures = (False or reloadData)
start = time.time()
(X,y) = load_features(data, forceReload = recalculateFeatures)
print("Loading features took {:.2f}s".format(time.time()-start))

'''
SPLITTING DATASETS
'''
from sklearn.cross_validation import train_test_split

X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.2, random_state=random_state)

'''
TRAIN DECISION TREE
'''
from decisiontree import report_result
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, grid_search
start = time.time()
rf = RandomForestClassifier()
param_grid = {"max_depth": [3, 4, 5],
              "max_features": [3, 5],
              "min_samples_split": [20, 40],
              "min_samples_leaf": [40, 80],
              "bootstrap": [True],
              "criterion": ["gini"]}
cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=50,test_size=0.2, random_state=123)
clf = grid_search.GridSearchCV(rf, param_grid, cv=cv, n_jobs=4, scoring='roc_auc', verbose=1)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_val)
report_result(y_pred,y_val)
print("Learning trainer took {:.2f}s".format(time.time()-start))

'''
RUN ON TEST DATA
'''
from load_data import load_test_data
(testdata,testfilepath) = load_test_data(maxRows=0)

X_test = load_features(testdata, traindata=False, forceReload = True)
del testdata
#X_test = testdata.drop(list(set(testdata.columns) - set(selected_features)),axis=1)
#X_test.d_price = X_test.d_price.fillna(0)

#y_test = predict_result(clf, X_test)


y_prob = clf.predict_proba(X_test)


'''
FINISH AND CREATE SUBMIT FILE
'''
print("Creating submssion file")
from save_data import create_submit_file_kaggle_avito
create_submit_file_kaggle_avito(y_prob)

