# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 19:49:31 2016

@author: guusjeschouten
"""

from sklearn import tree
from sklearn import metrics
from sklearn import grid_search
import time
from sklearn.externals.six import StringIO 
import pydot

def predict_using_decision_tree(X_train, y_train, X_test, forceUpdateClf = False):

    clf = train_decision_tree(X_train, y_train, forceUpdateClf)
    
    return predict_result(clf, X_test)

def optimized_tree_parameters(X_train, y_train, forceUpdateClf = False):
    
    if not forceUpdateClf: 
        return {"criterion": "gini",
                      "min_samples_split": 2,
                      "max_depth": 5,
                      "min_samples_leaf": 20,
                      "max_leaf_nodes": None,
                      }
    
    param_grid = {"criterion": ["gini", "entropy"],
                  "min_samples_split": [2, 10, 20],
                  "max_depth": [None, 2,3,4,5],
                  "min_samples_leaf": [10, 20, 30],
                  "max_leaf_nodes": [None],
                  }
              #
              #
    
    tr = tree.DecisionTreeClassifier()
    
    
    start = time.time()
    clf = grid_search.GridSearchCV(tr, param_grid, cv=5)
    clf = clf.fit(X_train,y_train)
    print("Optimizing decision tree took {:.2f}s".format(time.time()-start))
    print "With parameters:"

    best_parameters = clf.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name]) 
    
    return clf.best_params_

    
def train_decision_tree(X_train, y_train, forceUpdateClf = False):
    
    params = optimized_tree_parameters(X_train, y_train, forceUpdateClf)
    
    #print params
    
    clf = tree.DecisionTreeClassifier(**params)
    start = time.time()
    clf.fit(X_train,y_train)
    print("Training decision tree took {:.2f}s".format(time.time()-start))


    return clf
    
def predict_result(clf, X_test):
    start = time.time()
    y_res = clf.predict(X_test)
    print("Predicting using decision tree took {:.2f}s".format(time.time()-start))
    return y_res
    
    
def report_result(y_test, y_predict):
    print "Report DECISION TREE:"
    print "AUC Score: {:.2f}".format(metrics.roc_auc_score(y_test,y_predict))
    #print "Score: {:.2f}".format(clf.score(X_test,y_test))

    print "Report"    
    print metrics.classification_report(y_test,y_predict,target_names=['No duplicate','Duplicate'])

def save_image_decision_tree(clf, feature_names):

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names, class_names=['No duplicate','Duplicate'])
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("tree.png")