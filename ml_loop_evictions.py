import pipeline_evictions as pipeline

import pandas as pd
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
            #Set parameters to the model. ** alows us to use keyword arguments
            model.set_params(**p)

            #Train model
            model.fit(train_test_set['x_train'], train_test_set['y_train'])
            
            #Predict
            y_pred_scores=0
            if(models_to_run[index] == 'SVM'):
              y_pred_scores = model.decision_function(train_test_set['x_test'])
            else:
              y_pred_scores = model.predict_proba(train_test_set['x_test'])[:,1]
            

            #Sort according to y_pred_scores, keeping map to their y_test values
            y_pred_scores_sorted, y_test_sorted = zip(*sorted(zip(y_pred_scores, train_test_set['y_test']), reverse=True))

            #Define different thresholds to calculate metrics
            thresholds_for_metrics = [1,2,5,10,20,30,50]

            #Baseline will be precision at 100% (assign 1 to everybody)
            baseline = pipeline.metric_at_k(y_test_sorted,y_pred_scores_sorted,100,'precision')

            #Get precision, recall and f1 for the different thresholds
            prec_rec_f1 = pipeline.generate_precision_recall_f1(y_test_sorted,y_pred_scores_sorted, thresholds_for_metrics)

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