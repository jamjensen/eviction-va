
import pandas as pd 
import numpy as np
import os
from sklearn.model_selection import train_test_split


FEATURES = ['year','poverty-rate','median-gross-rent','eviction-filing-rate','evictions']

TARGET = ['top_10_eviction_rate']

PATH = 'data/tracts.csv'
TRAIN_YEAR = 2012
TEST_YEAR = 2013

class process:
    '''
    Class for data pipeline
    '''
    def __init__(self):

        self.df = self.load_data()

    def load_data(self):

        if os.path.exists(PATH):
            df = pd.read_csv(PATH)
        else:
            raise Exception('The file does not exist')
         
        df.drop('GEOID', inplace=True, axis=1)   

        return df

def make_target(df):

    df[TARGET] = np.where(df['eviction-rate'] >= df['eviction-rate'].quantile(.9), 1,0)
    # df['top_10_eviction_rate'] = np.where(df['eviction-rate'] >= df['eviction-rate'].quantile(.9), 1,0) 

    return df


def get_train_test_splits(df):

    df_train = df[df['year'] == TRAIN_YEAR]
    df_test = df[df['year'] == TEST_YEAR]

    x_train = df_train.loc[:,FEATURES]
    y_train = make_target(df_train)[TARGET]

    x_test = df_test.loc[:,FEATURES]
    y_test = make_target(df_test)[TARGET]

    return x_train, y_train, x_test, y_test




