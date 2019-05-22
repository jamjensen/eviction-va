from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
import pipeline



EMPTY_DF = pd.DataFrame(columns=('model_type','clf', 'parameters','baseline', 'auc-roc',
                                            'f1_at_5','f1_at_10','a_at_5', 'a_at_10',
                                            'p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50', 'r_at_1', 'r_at_2','r_at_5',
                                            'r_at_10', 'r_at_20','r_at_30','r_at_50'))


def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': LinearSVC(random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'BG': BaggingClassifier(LogisticRegression(penalty='l1', C=1e5))
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BG': {'n_estimators' : [10, 20], 'max_samples' : [.25, .5]}
           }

    
    test_grid = { 
    'RF':{'n_estimators': [1,5,10], 'max_depth': [1,5], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.01, .1]},
    # 'GB': {'n_estimators': [100, 50, 30], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10],'min_samples_split': [2,5,10]}
    # 'SVM' :{'C' :[0.01]},
    # 'KNN' :{'n_neighbors': [2,5,10],'weights': ['uniform'],'algorithm': ['auto']},
    # 'BG': {'n_estimators' : [2, 10, 20], 'max_samples' : [.1, .5]}
           }

    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return None


def clf_loop(models_to_run, clfs, grid, x_train, x_test, y_train, y_test):
"""Runs the loop using models_to_run, clfs, gridm and the data
"""
inner_df = EMPTY_DF.copy()

for n in range(1, 2):
    # create training and valdation sets
    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        print(models_to_run[index])
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            try:
                clf.set_params(**p)
                fit = clf.fit(x_train, y_train.values.ravel())
                if models_to_run[index] == 'SVM':
                    y_pred_probs = fit.decision_function(x_test)
                else:
                    y_pred_probs = fit.predict_proba(x_test)[:,1]
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test.values.ravel()), reverse=True))
                inner_df.loc[len(inner_df)] = [
                            models_to_run[index],clf, p,
                            baseline(y_test),
                           roc_auc_score(y_test, y_pred_probs),
                           f1_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                           f1_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                           accuracy_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                            accuracy_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                           precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                           precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                           precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                           precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                           precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                           precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                           precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                           recall_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                           recall_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                           recall_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                           recall_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                           recall_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                           recall_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                           recall_at_k(y_test_sorted,y_pred_probs_sorted,50.0)]
                           
                # plot_precision_recall_n(y_test,y_pred_probs,clf)
            except IndexError as e:
                print('Error:',e)
                continue

return inner_df