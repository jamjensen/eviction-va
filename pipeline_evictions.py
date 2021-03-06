# Based on https://github.com/rayidghani/magicloops

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import ParameterGrid

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

import textwrap

pd.options.mode.chained_assignment = None  # default='warn'


from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import datetime


def fill_na_columns_with_mean(df, columns_to_process):
  '''
  Filling in missing values of df with mean. Operate over specific columns only (columns_to_process)
  '''
  for col in columns_to_process:
        df[col].fillna(df[col].median(), inplace=True)  
  return df

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())

def impute_data(df, columns_to_process):
  '''
  Filling in missing values of df with mean. Operate over specific columns only (columns_to_process)
  '''
  for col in columns_to_process:

    # Fill by year
    by_year = df.groupby(['year'])

    df[col] = by_year[col].transform(lambda x: impute_median(x))

  return df



def create_temp_validation_train_and_testing_sets(df, data_column, label_column, split_thresholds, test_window, prediction_horizon, feature_generation_gap):
  '''
  Creates a series of temporal validation train and test sets
  Amount of train/test sets depends on length of split_thresholds array
  
  Training and test set are delimited by the split_thresholds
  data_column indicates which column of dataframe (df) shall be used to compare with split_threshold value

  features contain features of data
  label_colum indicates which column is the output label
  test_window indicates length of test data
  prediction_horizon indicates necessary time we need for train and test data to look at outcome (do not include data whose date_posted is in gap time hence)
  feature_generation_gap is the time of data needed to generate features. because one of the features is "top 10% last year", we cant use data from 2000 in training (we dont have 1999)
  '''

  #Array to save train and test sets
  train_test_sets=[None] * len(split_thresholds)

  #For each threshold, create training and test sets
  for index, split_threshold in enumerate(split_thresholds):

    #Save information of train and test set in dictionary
    train_test_set={}

    #The starting point of the test set will be the identification of this train/test sets
    train_test_set['test_set_start_date']=split_threshold

    #Columns of boolean values indicating if date_posted value is smaller/bigger than threshold
    
    #Train data is all data before threshold-gap. Keep gap for feature generation (at least one year over the minimum date)
    train_filter = (df[data_column] < split_threshold-prediction_horizon) & (df[data_column] >= df[data_column].min() + feature_generation_gap)

    #Test data is all data thats after training data(after split_threshold), but only consider a length of test_window time, - necessary gap to see all outcomes.
    test_filter = (df[data_column] >= split_threshold) & (df[data_column] < split_threshold+test_window-prediction_horizon)
    
    train_test_set['x_train'] = df.loc[:, df.columns != label_column][train_filter]
    train_test_set['y_train'] = df[label_column][train_filter]
    train_test_set['x_test'] = df.loc[:, df.columns != label_column][test_filter] 
    train_test_set['y_test'] = df[label_column][test_filter]
    
    train_test_sets[index]= train_test_set

  return train_test_sets

def get_best_models_of_each_type_for_each_train_test_set(models_to_run,results,train_test_set_column_identifier, metric_criteria):
  #In this dataframe we will save the best model for each type of model (ex 1 LR, 1 RF..), whichever perfomed the best in each train/test set
  best_models= pd.DataFrame()

  for model in models_to_run:
    #Filter data selecting only rows of this specific modelpd.to_numeric(
    results_of_model = results[results["model_name"]==model]  
    #For each train/test set, find index of best model (parameters)
    idx = results_of_model.groupby([train_test_set_column_identifier])[metric_criteria].transform(max) == results_of_model[metric_criteria]
      
    #Grab those results based on index
    best_model = results_of_model[idx]
    #Append it to final list
    best_models=best_models.append(best_model)

  return best_models

def create_features(train_test_set):

  #In the meantime
  train_features = train_test_set['x_train'].iloc[:,4:20] #Choosing first 20 columns in the meantime
  test_features = train_test_set['x_test'].iloc[:,4:20]

  return (train_features, test_features)

def get_models_and_parameters(grid='normal'):
  '''
  Get a set of classifiers and their possible parameters
  '''

  models = {

    'Baseline': 'Baseline model',
    'DT': DecisionTreeClassifier(random_state=0),
    'LR': LogisticRegression(penalty='l1', C=1),
    'RF': RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0),

    'BA': BaggingClassifier(KNeighborsClassifier(),n_estimators=10),
    'AB': AdaBoostClassifier(n_estimators=50),
    'GB': GradientBoostingClassifier(learning_rate=0.05, n_estimators=10, subsample=0.5),
    'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
    
    'SVM': LinearSVC(random_state=0, tol=1e-5, C=1, max_iter=10000),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'NB': GaussianNB()
  }

  normal_grid = { 

    'Baseline':{},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [2,10,50,100],'min_samples_split': [2,5]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,10]},#'C': [0.001,0.1,1,10]
    'RF': {'n_estimators': [100,10000], 'max_depth': [5,50,100], 'max_features': ['sqrt'],'min_samples_split': [2,10], 'n_jobs': [-1]},#'max_features': ['sqrt','log2']
    'BA': {'n_estimators': [10,100, 1000]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [10,100]},#'algorithm': ['SAMME', 'SAMME.R']
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1],'subsample' : [0.1,1.0]},
    'ET': { 'n_estimators': [100,100], 'criterion' : ['gini'] ,'max_depth': [2,5,50]},#'criterion' : ['gini', 'entropy']

    'SVM': {'C' :[10**-2, 10**-1, 1 , 10, 10**2]}, 
    'KNN': {'n_neighbors': [3,5,10,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree']},
    'NB' : {}
  }

  small_grid = { 

    'Baseline':{},
    'DT': {'criterion': ['gini'], 'max_depth': [2,50,100],'min_samples_split': [2]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.1,10]},
    'RF': {'n_estimators': [100,1000], 'max_depth': [5,50,100], 'max_features': ['sqrt'],'min_samples_split': [2], 'n_jobs': [-1]},
    'BA': {'n_estimators': [10,100]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [10,100]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1]},
    'ET': { 'n_estimators': [100,100], 'criterion' : ['gini'] ,'max_depth': [2,50]},

    'SVM': {'C' :[10**-2, 10**-1, 1 , 10, 10**2]}, 
    'KNN': {'n_neighbors': [3,5,10,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree']},
    'NB' : {}
  }
  

  test_grid = { 

    'Baseline':{},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'RF':{'n_estimators': [10], 'max_depth': [5], 'max_features': ['sqrt'],'min_samples_split': [10]},

    'BA': {'n_estimators': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},

    'SVM' :{'C' :[0.01]},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
    'NB' : {}

  }
  
  if(grid=='test'):
    return models, test_grid
  if(grid=='small'):
    return models, small_grid
  else:
    return models, normal_grid

def plot_models_in_time(models_to_run, best_models_df, metric):
  '''
  Plots how different models behave on metric during time
  models_to_run defines which models are to be ploted
  best_models_df contains the dataframe
  metric is the metric performance we are plotting
  '''

  #Clear plot
  plt.clf()

  #Create plot and axis
  fig, ax1 = plt.subplots()

  #Create lines for each model
  for model in models_to_run:
    ax1.plot( 'test_set_start_date', metric, data=best_models_df[best_models_df['model_name']==model], label=model, marker='o', markersize=6, linewidth=4,alpha=0.6)

  #Show legends
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

  #Set axis labels
  ax1.set_xlabel('Test set start date')
  ax1.set_ylabel(metric)

  ax1.set_ylim([0,1])
  
  plt.xticks(rotation=70)

  plt.show()

def joint_sort_descending(l1, l2):
    # Sort arrays descending together
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):   
    '''
    Generate values of 1 for y_scores in the top k%
    This method expects ordered y_scores
    ''' 

    #Find the index position where the top k% finishes
    cutoff_index = int(len(y_scores) * (k / 100.0))
    
    #Assign 1 to values in the top k %
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]

    return predictions_binary

def metric_at_k(y_true, y_scores, k, metric):
  '''
  Calculates metric given y_true and y_score values
  '''

  y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
  
  #generate binary y_scores
  binary_predictions_at_k = generate_binary_at_k(y_scores, k)


  # #classification_report returns different metrics for the prediction
  results = classification_report(y_true, binary_predictions_at_k, output_dict = True)
  

  if(metric=='precision'):
    metric = results['1.0']['precision']
  elif(metric=='recall'):
    metric = results['1.0']['recall']
  elif(metric=='f1'):
    metric = results['1.0']['f1-score']

  return metric

def generate_precision_recall_f1(y_test_sorted,y_pred_scores_sorted, thresholds):
  '''
  Calculates precision, recall and f1 metrics for different thresholds
  '''


  metrics = ['precision', 'recall', 'f1']

  output_array=[]

  for threshold in thresholds:
    for metric in metrics:
      metric_value = metric_at_k(y_test_sorted,y_pred_scores_sorted,threshold,metric)
      output_array.append(metric_value)
  return output_array

def plot_precision_recall_n(y_true, y_score, model, parameter_values, test_set_start_date, output_type='save'):

    '''
    Plot precision recall curves
    -y_true contains true values
    -y_score contains predictions
    -model is the model being run
    -parameter_values contains parameters used in this model: we will use this for the plot name
    -output_type: either saving plot or displaying
    '''



    #Compute precision-recall pairs for different probability thresholds
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score) 


    #The last precision and recall values are 1. and 0 in precision_recall_curve method, now removing them 
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]


    #We transform the pr_thresholds (which is an array with scores thresholds, to an array of percentage thresholds)
    pct_above_per_thresh = []
    number_scored = len(y_score)    
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)

    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    #Clear any existing figure
    plt.clf()

    #Create a figure and access to its axis
    fig, ax1 = plt.subplots()

    #Create blue line for precision curve
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')

    #Create a duplicate axis, and use it to plot recall curve
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
 
    #Limit axis borders
    ax1.set_ylim([0,1])
    ax1.set_xlim([0,1])
    ax2.set_ylim([0,1])
    
    #Set name of plot 
    model_name = str(model).split('(')[0]
    chosen_params = str(parameter_values)
    plot_name = model_name+'-'+chosen_params+'-test_set_start_date:'+test_set_start_date


    #Set title and position in plot
    title = ax1.set_title(textwrap.fill(plot_name, 70))
    fig.tight_layout()
    fig.subplots_adjust(top=0.75)    

    #Save or show plot
    if (output_type == 'save'):
        plt.savefig('precision-recall curves/'+str(plot_name)+'.png')
    elif (output_type == 'show'):
        plt.show()
    plt.close()