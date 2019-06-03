'''NOTES TO CONSIDER FOR FULL MODEL'''
# Ensure training data is never less than 6 years
# Note: we note that the census data is updated only every 5 years
# Specifically, 2000-2004, 2005-2009, 2010, 2011-2016
import numpy as np
import pandas as pd


AGGREGATION = 'parent-location'

# eg. poverty rate by tract/county
ORIGINAL = ['poverty-rate', 'population', 'pct-white', 'pct-af-am',
            'pct-hispanic', 'pct-am-ind', 'pct-asian', 'pct-nh-pi',
            'pct-multiple', 'pct-other', 'year',
            'median-household-income', 'median-gross-rent',
            'eviction-filing-rate', 'eviction-filings',
            'median-property-value', 'evictions', 'rent-burden',
            'renter-occupied-households', 'pct-renter-occupied',
            'evictions', 'eviction-rate']

# eg. poverty rate above X by tract/county
PERCENTAGES = ['poverty-rate', 'pct-white', 'pct-af-am', 'rent-burden',
               'pct-renter-occupied', 'eviction-filing-rate']
THRESHOLDS = [1, 5, 10, 25, 50, 75]

# eg. median income PERCENT change over past X years by tract/county
ABSOLUTES = ['median-household-income', 'median-gross-rent',
             'eviction-filings', 'median-property-value', 'evictions',
             'renter-occupied-households', 'population']
YEARS = [1, 2, 5]
THRESHOLDS_ABSOLUTE = [-.5, -.3, -.1, .05, .1, .3, .5]
THRESHOLDS_PERCENT = [-50, -30, -10, 5, 10, 30, 50]






# eg. population X-ile by tract/county (decile, quartile)
DISCRETE_GROUPS = ['population', 'renter-occupied-households']
GROUPS = [4, 10]

# eg. population in top X-ile by tract/county (decile, quartile)
DISCRETE_GROUPS_THRESHOLD = ['population', 'evictions']
TOP_GROUPS = [1, 2, 3, 4]

# eg. county categories
ORIGINAL_CATEGORICAL = ['parent-location']
TOP_X_REST_OTHER = [20]

# eg. median rent over STATE average (find this value)
STATE_COMPARISON = ['median-household-income', 'median-gross-rent']



## must implement this first ##
def avg_continuous_by_county(df, features=ORIGINAL, aggregation=AGGREGATION):
    '''
    Creates features that aggregate each continuous variable at the county
    level for each row (tract).
    '''
    for var in features:
        county_total = df.groupby([aggregation, 'year'])[var].mean().rename('county_average_'
                       + var).reset_index()
        df = df.merge(county_total)

    return df


def absolute_binary(df, features=PERCENTAGES,
                    thresholds=THRESHOLDS):
    '''
    Creates binary features that determine whether a feature is above a
    specified threshold, first at the tract and then at the county level.

    Eg. poverty rate above X by tract? poverty rate above X by county?
    '''
    for var in features:
        for thresh in thresholds:
            df[var + '_above_' + str(thresh)] = np.where(df[var] >= thresh, 1, 0)

    return df


def change_over_years(df, features=ABSOLUTES, years=YEARS):
    '''
    Creates continuous features that specify the percent change in a given
    feature over the past X years, given a set of years to iterate over,
    first at the tract and then at the county level.
    '''
    for var in features:
        for year in years:
            df.sort_values(by=['GEOID', 'year'])
            df[var + '_percent_change_over_previous_' + str(year) + '_years'] =\
                df.groupby('GEOID')[var].pct_change(periods=year)

    return df


def percent_change_over_years(df, features=PERCENTAGES, years=YEARS):
    '''
    Creates continuous features that specify the change in a given
    feature over the past X years, given a set of years to iterate over,
    first at the tract and then at the county level.
    '''
    for var in features:
        for year in years:
            df.sort_values(by=['GEOID', 'year'])
            df[var + '_change_over_previous_' + str(year) + '_years'] =\
                df.groupby('GEOID')[var].diff(periods=year)

    return df



def create_features(df):

    df = avg_continuous_by_county(df, features=ORIGINAL, aggregation=AGGREGATION)
    
    df = absolute_binary(df, features=PERCENTAGES, thresholds=THRESHOLDS)
    county_features = ['county_average_' + var for var in PERCENTAGES]
    df = absolute_binary(df, features=county_features, thresholds=THRESHOLDS)
    
    df = change_over_years(df, features=ABSOLUTES, years=YEARS)
    county_features = ['county_average_' + var for var in ABSOLUTES]
    df = change_over_years(df, features=county_features, years=YEARS)
    
    changes = [var + '_percent_change_over_previous_' + str(year) + '_years' for var in ABSOLUTES for year in YEARS]
    df = absolute_binary(df, features=changes, thresholds=THRESHOLDS_ABSOLUTE)
    county_changes = ['county_average_' + var + '_percent_change_over_previous_' + str(year) + '_years' for var in ABSOLUTES for year in YEARS]
    df = absolute_binary(df, features=county_changes, thresholds=THRESHOLDS_ABSOLUTE)
    
    df = percent_change_over_years(df, features=PERCENTAGES, years=YEARS)
    county_features = ['county_average_' + var for var in PERCENTAGES]
    df = percent_change_over_years(df, features=county_features, years=YEARS)
    
    changes = [var + '_change_over_previous_' + str(year) + '_years' for var in PERCENTAGES for year in YEARS]
    df = absolute_binary(df, features=changes, thresholds=THRESHOLDS_PERCENT)
    county_changes = ['county_average_' + var + '_change_over_previous_' + str(year) + '_years' for var in PERCENTAGES for year in YEARS]
    df = absolute_binary(df, features=county_changes, thresholds=THRESHOLDS_PERCENT)

    return df

#STEP FINAL: drop GEOID and name from features list
