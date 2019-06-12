'''
Bias and Fairness Analysis Leveraging Aequitas

Citation: https://github.com/dssg/aequitas
'''
import pandas as pd
import seaborn as sns
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot
import warnings; warnings.simplefilter('ignore')
import matplotlib.pyplot as plt


RAW = 'df_bias.csv'
FEATURES = ['population', 'poverty-rate', 'median-gross-rent',\
            'median-household-income', 'pct-white', 'pct-af-am',\
            'pct-hispanic']
BINS = 5
LABELS = ['first_quintile', 'second_quintile', 'third_quintile',\
          'fourth_quintile', 'fifth_quintile']


def discretize_continuous_variable(df, feature, bins):
    '''
    Creates a new column in the dataframe containing a categorical variable
    to represent a specified continuous variable. Bins are set automatically.
    Drops the original variable from the dataset.

    Inputs:
        df (dataframe): a pandas dataframe
        feature (str): the name of a continuous variable
        bins (int or pandas function): the bin size (set automatically)

    Return:
        df (dataframe): dataframe with a new, discretized column
    '''
    df[feature + '_bins'] = pd.cut(df[feature], bins=5, labels=LABELS).astype(str)
    df.drop([feature], axis=1, inplace=True)

    return df


def bias_analysis(raw=RAW, features=FEATURES):
    '''
    Completes a comprehensive bias analysis covering groups, disparities,
    and fairness calculations as specified by Aequitas. Several plots are
    commented out but can be included as needed.

    Inputs:
        raw (str): output CSV containing scores and true labels
        features (lst): list of features of interest
    Return:
        grid (df): pandas dataframe representing fairness results
    '''
    df = pd.read_csv(RAW)
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1,\
                                               inplace=True)
    df = df[['population', 'poverty-rate', 'median-gross-rent',\
            'median-household-income', 'pct-white', 'pct-af-am', 'pct-hispanic',\
            'true_label', 'predicted_label']]
    df.rename(columns={'predicted_label':'score'}, inplace=True)
    df.rename(columns={'true_label':'label_value'}, inplace=True)
    for feature in features:
        df = discretize_continuous_variable(df, feature, 5)
    
    df['label_value'] = df['label_value'].astype(int)
    df['score'] = df['score'].astype(float)
    df[['population_bins', 'poverty-rate_bins', 'median-gross-rent_bins',\
        'median-household-income_bins', 'pct-white_bins', 'pct-af-am_bins',\
        'pct-hispanic_bins']].applymap(str)

    # create and evaluate confusion matrix
    g = Group()
    xtab, _ = g.get_crosstabs(df, attr_cols=['population_bins', 'poverty-rate_bins',\
                                             'median-gross-rent_bins',\
                                             'median-household-income_bins',\
                                             'pct-white_bins', 'pct-af-am_bins',\
                                             'pct-hispanic_bins'])
    absolute_metrics = g.list_absolute_metrics(xtab)
    xtab[[col for col in xtab.columns if col not in absolute_metrics]]
    tab = xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(2)

    # plot metrics across all population groups
    # aqp = Plot()
    # sns.set(font_scale=0.75)
    # p = aqp.plot_group_metric_all(xtab, metrics=['ppr', 'pprev', 'fnr', 'fpr'], ncols=4)

    # calculate disparities between groups
    b = Bias()
    bdf = b.get_disparity_predefined_groups(xtab, original_df=df, ref_groups_dict=\
              {'population_bins':'fifth_quintile',\
              'poverty-rate_bins':'first_quintile',\
              'median-gross-rent_bins':'fifth_quintile',\
              'median-household-income_bins':'fifth_quintile',\
              'pct-white_bins':'fifth_quintile',\
              'pct-af-am_bins':'first_quintile',\
              'pct-hispanic_bins': 'first_quintile'}, alpha=0.05,
              mask_significance=True)
    bdf[['attribute_name', 'attribute_value','fpr_ref_group_value','fpr_disparity']]

    # comparisons with the majority group
    majority_bdf = b.get_disparity_major_group(xtab, original_df=df,
                                               mask_significance=True)
    majority_bdf[['attribute_name', 'attribute_value','fpr_ref_group_value',\
                 'fpr_disparity']]

    # plot disparity metrics across population groups
    # aqp = Plot()
    # sns.set(font_scale=0.75)
    # aqp.plot_disparity(bdf, group_metric='fpr_disparity', attribute_name='pct-af-am_bins')

    # Fairness calculations
    f = Fairness()
    fdf = f.get_group_value_fairness(bdf)
    aqp = Plot()
    fpr_fairness = aqp.plot_fairness_group(fdf, group_metric='fpr', title=True)
    plt.savefig('fig1.png', dpi=400)

    # Sample fairness plot
    fpr_disparity_fairness = aqp.plot_fairness_disparity(fdf,
                                                         group_metric='fpr',
                                                         attribute_name='pct-af-am_bins')
    plt.tight_layout(pad=5)
    plt.rc('text', usetex=False)
    plt.savefig('fig2.png', dpi=400)
    fdf[['attribute_name', 'attribute_value', 'Supervised Fairness']]

    return fdf
