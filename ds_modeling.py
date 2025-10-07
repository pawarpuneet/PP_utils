""" Data modeling related utility functions"""
import numpy as np
import pandas as pd
from scipy.stats import t
from itertools import combinations, permutations
from math import factorial
from tabulate import tabulate
from PP_utils import data_plotting

def GridSearchCV_results(grid_search_model,cv,X,y, scorer_name='score', multi_metric_eval=False, ranks_list=None, plot_score_by_split=True, plot_n_splits=30, run_ttest=False):
    '''returns cols: params, rank_test*, mean_test*,std_test*
    For multi-metric evaluation, cols end with _scorer_name instead of _score
    returns pairwise_comp_df, df_test when run_ttest = True
    otherwise returns df_test'''
    df = pd.DataFrame(grid_search_model.cv_results_)
    if multi_metric_eval and scorer_name == 'score':
        print('Must provide a scorer name in case of multi metric evaluation')
        return
    sort_by_col = 'rank_test_' + scorer_name
    df = df.sort_values(by=[sort_by_col])
    df = df.set_index(
        df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
        ).rename_axis("param_comb")
    
    # filter columns to get score for each CV split for plotting and t-test
    model_scores = df.filter(regex=f"split\d*_test_{scorer_name}")
    # filter columns for the summary df
    summary_cols = [col for col in df.columns if 'params' in col or 'rank_test' in col or 'mean_test' in col or 'std_test' in col]
    df_test = df[summary_cols]

    if plot_score_by_split:
        # Let's plot scores vs CV split
        data_plotting.plot_GridSearchCV_scores_by_split(df,scorer_name,n_splits = plot_n_splits,ranks_list = ranks_list)

    if ranks_list:
        df_test = df_test[df_test[sort_by_col].isin(ranks_list)]
        index_list = [n-1 for n in ranks_list]
        
    if run_ttest:
        if ranks_list:
            pairwise_comp_df = GridSearchCV_pairwise_ttest(model_scores.iloc[index_list],cv,X,y)
        else:
            pairwise_comp_df = GridSearchCV_pairwise_ttest(model_scores,cv,X,y)
        print('''
    Pairwise t-test for the param combinations.
    Hypotheses: 
    Null H0: Population Mean of test scores of the model_1 and model_2 are equal 
    Alternative H1: Population Mean of test score of model_1 is greater than model_2 
                    implying that model_1 is better than model_2
    Significance level used = 0.05
            ''')
        return pairwise_comp_df

    return df_test


# helper function for GridSearchCV_compute_ttest
def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std

# helper function for GridSearchCV_compute_ttest
# These functions are sourced from sklearn's example
# "Statistical comparison of models using grid search"
def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(t_stat, df)  # right-tailed t-test, sf = 1-cdf, 
    # sklearn example is using np.abs(t_stat), that is incorrect
    # we shouldn't use absolute value of t_stat for right tailed t-test
    return t_stat, p_val

def GridSearchCV_ttest(model_1_scores, model_2_scores,cv_generator,X,y):
    """Using paired t-test(Nadeau and Bengio’s corrected t-test), this function can be used to answer:
    Is the first model significantly better than the second model (when ranked by mean_test_score)?
    Assumption: both models have been tested on same set of KFolds.
    100 samples (i.e. 100 score values for each model on same splits) is a good number, 
    can be obtained by RepeatedKFold CV.
    cv_generator example: 
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
    X,y = data used for GridSearchCV -> training set, used for calculating n_train, n_test
    """

    differences = model_1_scores - model_2_scores
    n = differences.shape[0]  # number of test sets
    df = n - 1

    # number of samples in the training set and test set for each split in CV, 
    # get it from the first split
    n_train = len(next(iter(cv_generator.split(X, y)))[0])
    n_test = len(next(iter(cv_generator.split(X, y)))[1])

    t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
    result = 'reject_H0' if p_val < 0.05 else "don't reject H0"
    print(f"t-statistic: {t_stat:.3f}\n p-value: {p_val:.3f}\n result: {result}")
    print('''
        Hypotheses:
        Null H0: Mean test scores of both the models are equal 
        Alternative H1: Mean test score of model_1 is greater than model_2 
                        implying that model_1 is better than model_2
          ''')

def GridSearchCV_pairwise_ttest(model_scores,cv_generator,X,y):
    '''when there are only 2 models to compare, it gives the same result as GridSearchCV_ttest above'''
    n_comparisons = factorial(len(model_scores)) / (factorial(2) * factorial(len(model_scores) - 2))
    # number of samples in the training set and test set for each split in CV, 
    # get it from the first split
    n_train = len(next(iter(cv_generator.split(X, y)))[0])
    n_test = len(next(iter(cv_generator.split(X, y)))[1])

    pairwise_t_test = []

    for model_i, model_k in permutations(range(len(model_scores)), 2):
        model_i_scores = model_scores.iloc[model_i].values
        model_k_scores = model_scores.iloc[model_k].values
        differences = model_i_scores - model_k_scores
        n=differences.shape[0]
        df = n-1
        t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
        p_val *= n_comparisons  # implement Bonferroni correction
        # Bonferroni can output p-values higher than 1
        p_val = 1 if p_val > 1 else p_val
        pairwise_t_test.append(
            [model_scores.index[model_i], model_scores.index[model_k], t_stat, p_val]
        )

    pairwise_comp_df = pd.DataFrame(
        pairwise_t_test, columns=["model_1", "model_2", "t_stat", "p_val"]
    ).round(3)
    pairwise_comp_df['result'] = np.where(pairwise_comp_df['p_val'] < 0.05, 'reject H0', "don't reject H0")
    # print('''
    # Pairwise t-test for the param combinations.
    # Hypotheses:
    # Null H0: Mean test scores of both the models are equal 
    # Alternative H1: Mean test score of model_1 is greater than model_2 
    #                 implying that model_1 is better than model_2
    # Significance level used = 0.05
    #     ''')
    return pairwise_comp_df

