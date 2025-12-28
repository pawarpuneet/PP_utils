""" Feature Selection related utility functions"""
import pandas as pd
import numpy as np
from functools import partial
from sklearn.feature_selection import SelectKBest, mutual_info_classif

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
