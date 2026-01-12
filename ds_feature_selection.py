""" Feature Selection related utility functions"""
import pandas as pd
import numpy as np
from functools import partial
from collections import defaultdict
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

def clf_summarize_df_by_class(df, group_by_col, agg_dict=None, agg_func='mean'):
    """
    Group DataFrame by a column and apply aggregation.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        group_col (str): Categorical Column to group by (eg.target in clf problem)
        agg_dict (dict, optional): Dict mapping columns to functions. If None, use agg_func for all.
        agg_func (str or callable): Default function to apply if agg_dict not provided
    
    Returns:
        pd.DataFrame: Summarized DataFrame
    """
    if agg_dict is None:
        # Apply agg_func to all non-group columns
        cols_to_agg = [col for col in df.columns if col != group_by_col]
        agg_dict = {col: agg_func for col in cols_to_agg}
    
    return df.groupby(group_by_col, as_index=False).agg(agg_dict)   

def clf_select_top_features_using_mi(X, y, n, discrete_mask=None):
    """
    Select top n features using mutual_info_classif.
    mutual information methods can capture any kind of statistical dependency-linear or non-linear, 
    but being nonparametric, they require more samples for accurate estimation
    
    Parameters:
        X: DataFrame or array, shape (n_samples, n_features) can be mixed:cont and discrete
        y: array-like, shape (n_samples,) - target categorical
        n: int - number of features to select
        discrete_mask: bool array, optional - True for discrete features
    
    Returns:
        X_selected: array - transformed data
        selected_features: list - names of selected features (if X is DataFrame)
    """
    # Use 'auto' if no mask provided
    mi_func = partial(mutual_info_classif, discrete_features=discrete_mask) if discrete_mask is not None else mutual_info_classif
    
    selector = SelectKBest(score_func=mi_func, k=n)
    X_selected = selector.fit_transform(X, y)
    
    # Get feature names if input is a DataFrame
    selected_features = None
    if isinstance(X, pd.DataFrame):
        selected_features = X.columns[selector.get_support()].tolist()
    
    return selected_features

def select_features_from_clusters(dist_linkage, feature_names, threshold=1, selection_method='first'):
    """
    Select one representative feature from each hierarchical cluster.

    Parameters
    ----------
    dist_linkage : ndarray
        Linkage matrix from hierarchical clustering.
    feature_names : list or pd.Index
        Names of the features.
    threshold : float, default=1
        Distance threshold for forming flat clusters.
    selection_method : {'first', 'random'}, default='first'
        Strategy to select feature from each cluster.

    Returns
    -------
    selected_features : list
        Selected feature names.
    cluster_id_to_features : dict
        Mapping of cluster IDs to feature lists.

    Notes
    -----
    Use `data_plotting.plot_corr_dendrogram` to visualize the dendrogram, generate dist_linkage and
    determine an appropriate `threshold` value.
    To manually pick a threshold from a dendrogram:
    1. Look for large vertical distances between merges — these indicate distinct clusters.
    2. Draw a horizontal line across the dendrogram; the number of vertical lines it crosses 
    equals the number of clusters.
    3. Choose a threshold where the line cuts through the largest gaps, balancing cluster count 
    and separation.
    """
    cluster_ids = hierarchy.fcluster(dist_linkage, t=threshold, criterion="distance")
    cluster_id_to_features = defaultdict(list)
    
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_features[cluster_id].append(feature_names[idx])
    
    if selection_method == 'first':
        selected_features = [v[0] for v in cluster_id_to_features.values()]
    elif selection_method == 'random':
        import random
        selected_features = [random.choice(v) for v in cluster_id_to_features.values()]
    else:
        raise ValueError("selection_method must be 'first' or 'random'")
    
    return list(selected_features), dict(cluster_id_to_features)   
