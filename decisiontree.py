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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation

def predict_using_decision_tree(X_train, y_train, X_test, forceUpdateClf = False):

    clf = train_decision_tree(X_train, y_train, forceUpdateClf)
    
    return clf, predict_result(clf, X_test)

def optimized_tree_parameters(X_train, y_train, forceUpdateClf = False):
    
    if not forceUpdateClf: 
        return {"criterion": "gini",
                      "min_samples_split": 10,
                      "max_depth": 5,
                      "min_samples_leaf": 40,
                      "max_leaf_nodes": None,
                      }
    
    param_grid = {"criterion": ["gini", "entropy"],
                  "min_samples_split": [10, 20],
                  "max_depth": [2,3,4,5],
                  "min_samples_leaf": [20, 40],
                  "max_leaf_nodes": [None],
                  }
              #
              #
    
    tr = tree.DecisionTreeClassifier()
    
    
    start = time.time()
    cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=100,test_size=0.2, random_state=123)
    clf = grid_search.GridSearchCV(tr, param_grid, cv=cv, n_jobs=4, scoring='roc_auc')
    clf = clf.fit(X_train,y_train)
    
    #plot_learning_curve(clf.best_estimator_, "Testje",X_train, y_train, cv=cv, n_jobs=4)    
    
    print("Optimizing decision tree took {:.2f}s".format(time.time()-start))
    print("Best score:{} with scorer {}".format(clf.best_score_, clf.scorer_))
    print "With parameters:"

    best_parameters = clf.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name]) 
        
    save_image_decision_tree(clf.best_estimator_, X_train.columns)
    
    #return (clf.best_estimator_, clf.best_params_)
    return clf.best_params_

    
def train_decision_tree(X_train, y_train, forceUpdateClf = False):
    
    params = optimized_tree_parameters(X_train, y_train, forceUpdateClf)
    
    #print params
    
    clf = tree.DecisionTreeClassifier(**params)
    #print clf
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
   

    print "Report"    
    print metrics.classification_report(y_test,y_predict,target_names=['No duplicate','Duplicate'])

def save_image_decision_tree(clf, feature_names):

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names, class_names=['No duplicate','Duplicate'])
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("tree.png")
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt