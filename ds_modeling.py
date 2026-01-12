""" Data modeling related utility functions"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn.metrics import classification_report, class_likelihood_ratios, make_scorer, accuracy_score
from sklearn.model_selection import cross_validate, TunedThresholdClassifierCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances
from itertools import combinations, permutations
from math import factorial
from tabulate import tabulate
from PP_utils import data_plotting
import warnings
from sklearn.exceptions import UndefinedMetricWarning

##############################################################
################## Grid Search Utilities ##################
##############################################################

def GridSearchCV_results(grid_search_model,cv=None,X=None,y=None, scorer_name='score', multi_metric_eval=False, ranks_list=None, plot_score_by_split=True, plot_n_splits=30, run_ttest=False):
    '''returns cols: params, rank_test*, mean_test*,std_test*
    For multi-metric evaluation, cols end with _scorer_name instead of _score
    returns pairwise_comp_df, df_test when run_ttest = True
    otherwise returns df_test
    cv, X, y are only needed if run_ttest = True
    grid_search_model needs to have at least 100 samples for each model if run_ttest = True.
    '''
    
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

######### Following 4 functions related to ttest are sourced from sklearn's example ##########
##############################################################################################
# "Statistical comparison of models using grid search"
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
    pairwise_comp_df['result'] = np.where(pairwise_comp_df['p_val'] < 0.1, 'reject H0', "don't reject H0")
    # print('''
    # Pairwise t-test for the param combinations.
    # Hypotheses:
    # Null H0: Mean test scores of both the models are equal 
    # Alternative H1: Mean test score of model_1 is greater than model_2 
    #                 implying that model_1 is better than model_2
    # Significance level used = 0.1, if p_val < 0.1 then 'reject H0'
    #     ''')
    return pairwise_comp_df

def get_model_comparison_table(grid_search, metrics):
    """
    Extract and tabulate cross-validation results from GridSearchCV for multiple metrics.
    
    This function processes the cv_results_ dictionary from a fitted GridSearchCV object
    and creates a summary DataFrame showing mean and standard deviation of test scores
    for each model configuration across specified metrics. The results are ranked by 
    the first metric in the list.
    
    Args:
        grid_search (GridSearchCV): Fitted GridSearchCV object with multiple metrics
        metrics (list): List of metric names (e.g., ['precision', 'recall'])
    
    Returns:
        pd.DataFrame: DataFrame with model parameters, mean ± std scores for each metric, 
                     and ranking, sorted by performance
    """
    # Extract all cross-validation results from GridSearchCV
    results = grid_search.cv_results_
    
    # Initialize dictionary to store comparison data
    data = {
        # Convert model parameters to string for readability
        'Model': [str(p) for p in results['params']]
    }
    
    # Extract mean and standard deviation for each requested metric
    for metric in metrics:
        data[f'Mean {metric.title()}'] = results[f'mean_test_{metric}']
        data[f'Std {metric.title()}'] = results[f'std_test_{metric}']
    
    df = pd.DataFrame(data)
    
    # Rank models by first metric (typically primary evaluation metric)
    df['Rank'] = df[f'Mean {metrics[0].title()}'].rank(ascending=False)
    
    # Sort by performance rank for easy interpretation
    return df.sort_values('Rank')

################### Functions for custom strategy for best estimator from GridSearchCV ###################
##########################################################################################################

def print_dataframe(filtered_cv_results,scorer_1, scorer_2):
    """Pretty print for filtered dataframe"""
    for mean_1, std_1, mean_2, std_2, params in zip(
        filtered_cv_results[f"mean_test_{scorer_1}"],
        filtered_cv_results[f"std_test_{scorer_1}"],
        filtered_cv_results[f"mean_test_{scorer_2}"],
        filtered_cv_results[f"std_test_{scorer_2}"],
        filtered_cv_results["params"],
    ):
        print(
            f"{scorer_1}: {mean_1:0.3f} (±{std_1:0.03f}),"
            f" {scorer_2}: {mean_2:0.3f} (±{std_2:0.03f}),"
            f" for {params}"
        )
    print()

def custom_strategy_for_best_estimator(cv_results, scorer_1, scorer_2, threshold_scorer_1):
    """Define the strategy to select the best estimator.
    This won't work directly with refit parameter of GridSearchCV.
    But can be modified by hardcoding threshold_scorer_1, scorer_1 and scorer_2 names 
    and removing those from the function input parameters.

    The strategy is to short-list the models which are the best in terms of any two scoring metrics
    used in GridSearchCV. And then choose fastest amongst those.
    Step1: We use a threshold value to filter models for metric_1 and 
    Step2: we filter for metric_2 by choosing models within 1 stdev of max of metric_2. 
    Step3: we apply the final filter to get the fastest of all.

    Parameters
    ----------
    cv_results : dict of numpy (masked) ndarrays
        CV results as returned by the `GridSearchCV`.

    Returns
    -------
    best_index : int
        The index of the best estimator as it appears in `cv_results`.
    """
    # print the info about the grid-search for the different scores
    cv_results_ = pd.DataFrame(cv_results)
    print("All grid-search results:")
    print_dataframe(cv_results_, scorer_1, scorer_2)

    # Filter-out all results below the scorer_1 threshold
    high_scorer_1_cv_results = cv_results_[
        cv_results_[f"mean_test_{scorer_1}"] > threshold_scorer_1
    ]

    print(f"Models with a {scorer_1} higher than {threshold_scorer_1}:")
    print_dataframe(high_scorer_1_cv_results, scorer_1, scorer_2)

    # Select columns: params, mean_score_time
    # mean_test_<scorer_name>, std_test_<scorer_name>,
    # rank_test_<scorer_name>
    col_regex = f'(params|mean_score_time|(mean|std|rank)_test_({scorer_1}|{scorer_2}))'

    high_scorer_1_cv_results = high_scorer_1_cv_results.filter(regex = col_regex) # default df.filter is col filter

    # Select the most performant models in terms of scorer_2
    # (within 1 sigma from the best)
    best_scorer_2_std = high_scorer_1_cv_results[f"mean_test_{scorer_2}"].std()
    best_scorer_2 = high_scorer_1_cv_results[f"mean_test_{scorer_2}"].max()
    best_scorer_2_threshold = best_scorer_2 - best_scorer_2_std

    high_scorer_2_cv_results = high_scorer_1_cv_results[
        high_scorer_1_cv_results[f"mean_test_{scorer_2}"] > best_scorer_2_threshold
    ]
    print(
        f"Out of the previously selected high {scorer_1} models, we keep all the\n"
        f"the models within one standard deviation of the highest {scorer_2} model:"
    )
    print_dataframe(high_scorer_2_cv_results, scorer_1, scorer_2)

    # From the best candidates, select the fastest model to predict
    fastest_top_scorer_2_high_scorer_1_index = high_scorer_2_cv_results[
        "mean_score_time"
    ].idxmin()

    print(
        "\nThe selected final model is the fastest to predict out of the previously\n"
        f"selected subset of best models based on {scorer_1} and {scorer_2}.\n"
        "Its scoring time is:\n\n"
        f"{high_scorer_2_cv_results.loc[fastest_top_scorer_2_high_scorer_1_index]}"
    )

    return fastest_top_scorer_2_high_scorer_1_index

# custom strategy for balancing model complexity and cross-validated score from GridSearchCV #
##############################################################################################
# Function 'best_low_complexity' can be directly used with the refit parameter of GridSearchCV
# by finding a decent accuracy within 1 standard deviation of the best accuracy score 
# while minimising the number of PCA components
# Following two functions ONLY work when SINGLE scoring metric is used for GridSearchCV
# and this method requires PCA to be part of the pipeline named "reduced_dim", example:
# pipe = Pipeline(
#     [
#         ("reduce_dim", PCA(random_state=42)),
#         ("classify", LogisticRegression(random_state=42, C=0.01, max_iter=1000)),
#     ]
# )
# Use a non-stratified CV strategy to make sure that the inter-fold
# standard deviation of the test scores is informative.
# Refer: Balance model complexity and cross-validated score on sklearn under examples
def lower_bound(cv_results):
    """
    Calculate the lower bound within 1 standard deviation
    of the best `mean_test_scores`.

    Parameters
    ----------
    cv_results : dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`

    Returns
    -------
    float
        Lower bound within 1 standard deviation of the
        best `mean_test_score`.
    """
    best_score_idx = np.argmax(cv_results["mean_test_score"])

    return (
        cv_results["mean_test_score"][best_score_idx]
        - cv_results["std_test_score"][best_score_idx]
    )


def best_low_complexity(cv_results):
    """
    Balance model complexity with cross-validated score.

    Parameters
    ----------
    cv_results : dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`.

    Return
    ------
    int
        Index of a model that has the fewest PCA components
        while has its test score within 1 standard deviation of the best
        `mean_test_score`.
    """
    threshold = lower_bound(cv_results)
    candidate_idx = np.flatnonzero(cv_results["mean_test_score"] >= threshold)
    best_idx = candidate_idx[
        cv_results["param_reduce_dim__n_components"][candidate_idx].argmin()
    ]
    return best_idx

##############################################################
################## Classification Utilities ##################
##############################################################

### following two are helper functions for classification_results ###

def scoring(estimator, X, y):
    y_pred = estimator.predict(X)
    pos_lr, neg_lr = class_likelihood_ratios(y, y_pred, replace_undefined_by=1.0)
    return {"positive_likelihood_ratio": pos_lr, "negative_likelihood_ratio": neg_lr}

def extract_score(cv_results):
    lr = pd.DataFrame(
        {
            "positive": cv_results["test_positive_likelihood_ratio"],
            "negative": cv_results["test_negative_likelihood_ratio"],
        }
    )
    return lr # lr.aggregate(["mean", "std"])

def classification_results(X,y,estimator, y_pred=None, classes=None):
    '''
    calculates predicted from estimator if specified and
    prints the following:
    1. classification report
    2. class likelihood ratios
    
    Parameters
    ----------
    X : matrix like 
    y : target 
    estimator : classifier model fitted
    y_pred : predicted target (optional, needed if no estimator is provided)
    classes : list of class labels (optional, needed if no estimator is provided)

    Returns
    -------
    None
    '''
    if estimator:
        y_pred = estimator.predict(X)
        classes = estimator.classes_
    else:
        if y_pred is None or classes is None:
            print('Must provide y_pred and classes (list of class labels) if no estimator is provided.')
            return
    classes = [str(label) for label in classes]
    print("Calssification Report")
    print("---------------------")    
    print(classification_report(y, y_pred, target_names=classes))
    pos_LR, neg_LR = class_likelihood_ratios(y, y_pred, replace_undefined_by=1.0)
    print("Class Likelihood Ratios")
    print("-----------------------")
    print(f"LR+: {pos_LR:.3f}, LR-: {neg_LR:.3f}")
    print(f'''The post-test odds that the condition is truly present given a positive test result 
          are {pos_LR:.3f} times the pre-test odds.
          
          LR+/- close to 1.0  : classifier is NOT better than a dummy model 
                               that will output random predictions 
                               or always predict the most frequent class.
          LR+ > 1 and LR- close to 0: classifier is better than a dummy model
          ''')

def classification_cross_validate_likelihood_ratios(X,y,estimator, n_splits = 10):
    print("Cross Validation of Likelihood ratios")
    print("-------------------------------------")
    lr = extract_score(cross_validate(estimator, X, y, scoring=scoring, cv=n_splits))
    print(lr,lr.aggregate(['mean','std']), sep='\n')

    print(f'''          
          LR+/- close to 1.0  : classifier is NOT better than a dummy model 
                               that will output random predictions 
                               or always predict the most frequent class.
          LR+ > 1 and LR- close to 0: classifier is better than a dummy model
          ''')
    print(f'''Regarding UndefinedMetricWarning: If `positive_likelihood_ratio` is ill defined for certain splits, the value is nan. 
    np.nan is replaced by 1.0 (replace_undefined_by=1.0) as a way to denote poor performance of the classifier.\n''')

def classification_tuning_decision_threshold(X_train,y_train,X_test,y_test,base_model, scoring_metric, pos_label=1, multiclass=True):
    '''By default TunedThresholdClassifierCV uses a 5-fold stratified cross-validation to tune the decision threshold. 
    The parameter cv allows to control the cross-validation strategy'''
    if multiclass:
        label_binarizer = LabelBinarizer().fit(y_train)
        y_onehot_train = label_binarizer.transform(y_train)
        y_onehot_test = label_binarizer.transform(y_test)
        class_id = np.flatnonzero(label_binarizer.classes_ == pos_label)[0]
        y_train_binary = y_onehot_train[:, class_id]
        y_test_binary = y_onehot_test[:, class_id]
        pos_label=1 # with binarizing, the class of interest is labeled as 1
    else:
        y_train_binary = y_train
        y_test_binary = y_test
    if scoring_metric == accuracy_score:
        scorer = make_scorer(scoring_metric) 
    else:
        scorer = make_scorer(scoring_metric, pos_label=pos_label) 
    model = TunedThresholdClassifierCV(base_model, scoring=scorer)
    score_without_tuning = scorer(base_model.fit(X_train,y_train_binary), X_test, y_test_binary)
    score_with_tuning = scorer(model.fit(X_train, y_train_binary), X_test, y_test_binary)
    best_score_cv = model.best_score_
    print(f'''
    Score (test set) before tuned threshold = {score_without_tuning}\n
    Score (test set) with tuned threshold = {score_with_tuning}\n
    Best Score from cv = {best_score_cv}''')

def classification_tuning_decision_threshold_binary(X_train,y_train,X_test,y_test,base_model, scoring_metric, pos_label=1):
    '''By default TunedThresholdClassifierCV uses a 5-fold stratified cross-validation to tune the decision threshold. 
    The parameter cv allows to control the cross-validation strategy'''

    if scoring_metric == accuracy_score:
        scorer = make_scorer(scoring_metric) 
    else:
        scorer = make_scorer(scoring_metric, pos_label=pos_label) 
    model = TunedThresholdClassifierCV(base_model, scoring=scorer)
    score_without_tuning = scorer(base_model.fit(X_train,y_train), X_test, y_test)
    score_with_tuning = scorer(model.fit(X_train, y_train), X_test, y_test)
    best_score_cv = model.best_score_
    print(f'''
    Score (test set) before tuned threshold = {score_without_tuning}\n
    Score (test set) with tuned threshold = {score_with_tuning}\n
    Best Score from cv = {best_score_cv}''')

##############################################################
################## Clustering Utilities ##################
##############################################################

def dunn_index(labels, distances):
    """
    Calculate the Dunn Index for cluster validation (higher is better).
    Dunn Index: Measures cluster compactness and separation.
    It is calculated as the ratio of the smallest inter-cluster distance 
    to the largest intra-cluster diameter.
    Used in clustering_performance_evaluation
    """
    labels = np.array(labels)
    distances = np.array(distances)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Compute diameter of each cluster (max intra-cluster distance)
    diameters = []
    for label in unique_labels:
        in_cluster = distances[labels == label][:, labels == label]
        diameters.append(np.max(in_cluster) if in_cluster.size > 0 else 0)
    max_diameter = np.max(diameters)
    
    # Compute minimal inter-cluster distance
    inter_distances = []
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:
                between = distances[labels == label_i][:, labels == label_j]
                if between.size > 0:
                    inter_distances.append(np.min(between))
    min_inter_distance = np.min(inter_distances)
    
    return min_inter_distance / max_diameter

def clustering_performance_evaluation(X, labels_dict,plot_silhouette_analysis=True):
    """
    Evaluate and compare clustering performance across multiple models with a table output.
    
    Parameters:
    X : array-like of shape (n_samples, n_features)
        The input data.
    labels_dict : dict
        Dictionary with model names as keys and cluster labels as values.
    
    Silhouette Coefficient (range: -1 to 1):
        A score close to 1 indicates that clusters are well-separated and samples are correctly assigned.
        A score near 0 suggests overlapping clusters.
        A negative score means some samples may be assigned to the wrong cluster.
    Calinski-Harabasz Index:
        A higher value indicates better clustering, with dense and well-separated clusters.
        This index compares between-cluster variance to within-cluster variance—larger values mean 
        stronger separation.
    Davies-Bouldin Index:
        A lower value (closer to 0) indicates better clustering.
        It measures the average similarity between each cluster and its most similar one—
        smaller values mean clusters are more distinct and compact
    Dunn Index: 
        Measures cluster compactness and separation.higher is better
        It is calculated as the ratio of the smallest inter-cluster distance 
        to the largest intra-cluster diameter.
    """
    if plot_silhouette_analysis:
        data_plotting.clustering_silhouette_analysis(X,labels_dict)
    results = []
    distances = euclidean_distances(X)
    
    for model_name, labels in labels_dict.items():
        sil_score = silhouette_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        dunn_score = dunn_index(labels, distances)
        
        results.append({
            'Model': model_name,
            'Silhouette': f"{sil_score:.3f}",
            'Calinski-Harabasz': f"{ch_score:.3f}",
            'Dunn Index': f"{dunn_score:.3f}",
            'Davies-Bouldin': f"{db_score:.3f}"
        })
    
    # Display results
    df = pd.DataFrame(results)
    #print(df.to_string(index=False))

    print("\n--- Interpretation ---")
    print("• Silhouette Coefficient: Higher is better (close to 1).")
    print("• Calinski-Harabasz Index: Higher is better.")
    print("• Dunn Index: Higher is better.")
    print("• Davies-Bouldin Index: Lower is better (closer to 0).")    
    # Apply color formatting and return
    return df.style \
        .background_gradient(cmap='Greens', subset=['Silhouette', 'Calinski-Harabasz', 'Dunn Index']) \
        .background_gradient(cmap='Greens_r', subset=['Davies-Bouldin'])

##############################################################
################## Generic Model Inspection Utilities ##################
##############################################################
def plot_permutation_importance(estimator, X, y, n_repeats=10, random_state=42, title="Permutation Importances",scoring_metric=None):
    """
    Plot permutation feature importance for a given estimator and dataset.
    Use it on both training and test set. The difference between those two plots
    can tell us if model is using any features to overfit.
    We can then remedy the overfit and then if we try it on training and test set
    feature importance should look similar.

    If we have high scoring model but no feature appears important, we have multicollinearity in our data.
    Model is getting info from other correlated features, so no big drop in scoring metric.
    Once we keep only one of the highly correlated features, we should see drop in scoring metric.

    Read more here: 
    https://scikit-learn.org/stable/modules/permutation_importance.html
    https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
    Parameters
    ----------
    estimator : object
        A fitted sklearn-compatible estimator.
    X : pd.DataFrame or array-like
        Feature data (test or validation set).
    y : array-like
        Target values.
    n_repeats : int, default=10
        Number of times to permute each feature.
    random_state : int, default=42
        Random seed for reproducibility.
    title : str, default="Permutation Importances"
        Title for the plot.
    scoring_metric: Scorer to use, if None, the estimator’s default evaluation criterion is used.
    Returns
    -------
    result : Bunch
        Result from sklearn's permutation_importance.
    ax : matplotlib.axes.Axes
        Axes object of the generated boxplot.
    """
    result = permutation_importance(
        estimator, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1, scoring=scoring_metric
    )
    sorted_idx = result.importances_mean.argsort()
    importances_df = pd.DataFrame(
        result.importances[sorted_idx].T,
        columns=X.columns[sorted_idx]
    )
    ax = importances_df.plot.box(vert=False, whis=10)
    ax.set_title(title)
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in score")
    ax.figure.tight_layout()
    print("Note: Permutation importance does not reflect the intrinsic predictive value of " \
    "a feature by itself but how important this feature is for a particular model.")
    return result, ax   

def plot_feature_importance(model, model_name, feature_names):
    """
    Extracts and plots feature importance for Tree-based models and regression.
    """
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'): # For  Regression
        importances = np.abs(model.coef_[0])
    
    if importances is not None:
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 5))
        plt.title(f"Feature Importance - {model_name}")
        plt.bar(range(len(importances)), importances[indices], align="center", color='teal')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print(f"{model_name} does not provide direct feature importance.")

