import pipeline_evictions as pipeline

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

from sklearn.model_selection import ParameterGrid


def iterate_over_models_and_training_test_sets(models_to_run, models, parameters_grid, train_test_sets):
  '''
  Iterates over a variety of classifiers, parameters and training/test sets
  '''
  results_df =  pd.DataFrame(columns=(
    'model_name',
    'model',
    'parameters',
    'test_set_start_date',
    'baseline',
    'p_at_1',
    'r_at_1',
    'f1_at_1',
    'p_at_2',
    'r_at_2',
    'f1_at_2',
    'p_at_5',
    'r_at_5',
    'f1_at_5',
    'p_at_10',
    'r_at_10',
    'f1_at_10',
    'p_at_20',
    'r_at_20',
    'f1_at_20',
    'p_at_30',
    'r_at_30',
    'f1_at_30',
    'p_at_50',
    'r_at_50',
    'f1_at_50',
    'auc-roc'))

  #For each training and test set
  for train_test_set in train_test_sets:

  # For each of our models
    for index,model in enumerate([models[x] for x in models_to_run]):

      #Get all possible parameters for the current model
      parameter_values = parameters_grid[models_to_run[index]]


      #For every combination of parameters
      for p in ParameterGrid(parameter_values):
        print(str(datetime.now())+": Running "+str(models_to_run[index])+" with params: "+str(p) +" on train/test set "+str(train_test_set['test_set_start_date']))  
        try:

            #Baseline model starts assigning 1 to everybody
            y_pred_scores= np.ones(len(train_test_set['x_test']))

            if(models_to_run[index] != 'Baseline'):

              #Set parameters to the model. ** alows us to use keyword arguments
              model.set_params(**p)

              #Train model. Avoid selecting first column (geoid) from x_train to train model
              model.fit(train_test_set['x_train'].iloc[:,1:], train_test_set['y_train'])
              

              #Predict
              if(models_to_run[index] == 'SVM'):
                y_pred_scores = model.decision_function(train_test_set['x_test'].iloc[:,1:])
              else:
                #Get second column of predict_proba. Also removing first colum of test set (geoid)
                y_pred_scores = model.predict_proba(train_test_set['x_test'].iloc[:,1:])[:,1]
            
            else: #Baseline model
              #Baseline model predicts that the risky tracts are the same as those in top 10% last year
              y_pred_scores = train_test_set['x_test']['top_10_percent_last_year']



            #Define different thresholds to calculate metrics
            thresholds_for_metrics = [1,2,5,10,20,30,50]



            #Save df for bias & fairness analysis, + predictions
            df_bias = pd.DataFrame()

            if(models_to_run[index]=="LR"):

              #Copy all features
              df_bias = train_test_set['x_test'].copy(deep=True)
              #Copy true true_labelel
              df_bias['true_label'] = train_test_set['y_test'].copy(deep=True).reset_index(drop=True)
              #Copy score
              df_bias['score'] = y_pred_scores.copy()
              #Sort by score
              df_bias.sort_values(by ='score', ascending= False, inplace=True)

              predictions_at_10 = pipeline.generate_binary_at_k(df_bias['score'], 10)
              df_bias['predicted_label'] = predictions_at_10

              df_bias.to_csv('best_model_predictions.csv')


            #Baseline will be precision at 100% (assign 1 to everybody)
            baseline = pipeline.metric_at_k(train_test_set['y_test'],y_pred_scores,100,'precision')

            #Get precision, recall and f1 for the different thresholds
            prec_rec_f1 = pipeline.generate_precision_recall_f1(train_test_set['y_test'],y_pred_scores, thresholds_for_metrics)

            #Calculate roc metric
            roc_auc = pipeline.roc_auc_score(train_test_set['y_test'], y_pred_scores)

            #Define an identifier for this train/test sets. Will be the starting date of the test set
            test_set_identifier = str(train_test_set['test_set_start_date']).split(' ')[0]

            #Save results
            results_df.loc[len(results_df)] = [models_to_run[index],
                                               model,
                                               p,
                                               test_set_identifier,
                                               baseline
                                               ]+prec_rec_f1+[roc_auc]
            
            #Plot and save precision recall curve
            pipeline.plot_precision_recall_n(train_test_set['y_test'],y_pred_scores,model,p,str(train_test_set['test_set_start_date']),'save')


        except IndexError as e:
            print('Error:',e)

  return results_df