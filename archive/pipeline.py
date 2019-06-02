
import pandas as pd 
import numpy as np
import os
from sklearn.model_selection import train_test_split


FEATURES = ['poverty-rate','median-gross-rent','eviction-filing-rate','evictions']

TARGET = ['top_10_eviction_rate']

PATH = 'data/tracts.csv'
TRAIN_YEAR = 2012
TEST_YEAR = 2013


def load_data():
    df=None
    if os.path.exists(PATH):
        df = pd.read_csv(PATH)
    else:
        raise Exception('The file does not exist')
     
    # df.drop('GEOID', inplace=True, axis=1)

    return df

def make_target(df):

    df.loc[:,TARGET] = np.where(df['eviction-rate'] >= df['eviction-rate'].quantile(.9), 1,0)
    # df['top_10_eviction_rate'] = np.where(df['eviction-rate'] >= df['eviction-rate'].quantile(.9), 1,0)

    return df


def get_train_test_splits(df):

    df_train = df[df['year'] == TRAIN_YEAR]
    df_test = df[df['year'] == TEST_YEAR]

    x_train = df_train.loc[:,FEATURES]
    y_train = df_train.loc[:, TARGET]
    # y_train = make_target(df_train)[TARGET]

    x_test = df_test.loc[:,FEATURES]
    y_test = df_test.loc[:,TARGET]
    # y_test = make_target(df_test)[TARGET]
    # x_train.drop('year', inplace=True, axis=1)
    # x_test.drop('year', inplace=True, axis=1)

    return x_train, y_train, x_test, y_test


def fill_continuous_null(df, cols):
    '''
    Fills null columns in a dataframe in place
    Inputs:
        cols: list of column names corresponding to continuous variables
            with null values
    '''

    for col in cols:
        df[col].fillna(df[col].median(), inplace=True)

    return None


def discretize(df, colname, bin_len):
    '''
    Discretizes a continuous variable
    Inputs:
        df: a dataframe
        colname (str): name of continuous variable
        bin_len (int): size of bins 
    '''
    
    lb = df[colname].min()
    ub = df[colname].max()
    bins = np.linspace(lb, ub, bin_len)

    df[colname] = np.digitize(df[colname], bins)

    return df




