""" Data preprocessing related utility functions"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def onehot_encode(df, cat_cols=[], drop=None):
    """ Create a ColumnTransformer that applies OneHotEncoder to the specified columns
    and passes through the remaining columns

    Parameters
    ----------
    df : pd.DataFrame of shape (n_samples, n_features)
        Dataframe to be transformed
    cat_cols : list of categorical column names to be one hot encoded
        Number of samples in the training set.

    Returns
    -------
    df_transformed : pd.DataFrame
        transformed dataframe with one hot encoded cat_cols and remaining cols unchanged
    """
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop=drop), cat_cols)],
        remainder="passthrough", # Keeps non-categorical columns unchanged
        verbose_feature_names_out=False
    )

    # Apply the transformation to the DataFrame
    df_transformed_array = preprocessor.fit_transform(df)

    # Reconstruct the DataFrame using get_feature_names_out()
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    remaining_cols = [col for col in df.columns if col not in cat_cols]
    new_column_names = list(ohe_feature_names) + remaining_cols
    df_transformed = pd.DataFrame(df_transformed_array, columns=new_column_names)   

    # Convert the output to integers explicitly because 
    # ColumnTransformer concatenates outputs using np.hstack, 
    # which upcasts mixed dtypes to a common type (usually float64).
    # Even if OneHotEncoder(dtype=int) is used, numerical columns (typically float) 
    # force the entire output to float
    df_transformed[ohe_feature_names] = df_transformed[ohe_feature_names].astype(int)
    return df_transformed, ohe_feature_names

def onehot_encode_train_test(X_train,X_test, cat_cols=[], drop=None):
    """
    Apply OneHotEncoder to specified columns using fit on train and transform on both.
    
    Parameters:
        X_train (pd.DataFrame): Training data
        X_test (pd.DataFrame): Test data
        cat_cols (list): List of column names to one-hot encode
    
    Returns:
        X_train_encoded, X_test_encoded (tuple): Transformed DataFrames
    """
    # Create ColumnTransformer with OneHotEncoder for categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int,drop=drop), cat_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    # Fit on training data and transform both
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    
    # Get feature names and reconstruct DataFrames
    feature_names = preprocessor.get_feature_names_out()
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)
    
    # ColumnTransformer concatenates outputs using np.hstack, 
    # which upcasts mixed dtypes to a common type (usually float64).
    # Even if OneHotEncoder(dtype=int) is used, numerical columns (typically float) 
    # force the entire output to float
    # Identify categorical (one-hot) columns and convert to int
    cat_feature_names = [name for name in feature_names if name in preprocessor.named_transformers_['cat'].get_feature_names_out()]
    X_train_encoded[cat_feature_names] = X_train_encoded[cat_feature_names].astype(int)
    X_test_encoded[cat_feature_names] = X_test_encoded[cat_feature_names].astype(int)
    return X_train_encoded, X_test_encoded, cat_feature_names

def transform_cols_and_reconstuct_df(df, transformers=[]):
    """
    Transform specified columns of a DataFrame using a ColumnTransformer and return a new DataFrame.

    Applies a list of transformers to specified columns, leaves unspecified columns unchanged,
    and reconstructs the output into a pandas DataFrame with proper feature names.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to transform.
    transformers : list of tuples
        List of tuples specifying transformers and columns, formatted as (name, transformer, columns).
        Example: [('num', StandardScaler(), ['age', 'income'])].

    Returns
    -------
    df_encoded : pandas.DataFrame
        Transformed DataFrame with transformed and passthrough columns.
    feature_names : array of str
        Array of output feature names from the ColumnTransformer.

    Example
    -------
    >>> from sklearn.preprocessing import StandardScaler
    >>> transformers = [('num', StandardScaler(), ['age'])]
    >>> df_encoded, names = transform_cols_and_reconstuct_df(df, transformers)
    """
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    df_encoded = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names_out()
    df_encoded = pd.DataFrame(df_encoded, columns=feature_names, index=df.index)
    return df_encoded, feature_names   


def stratified_split(X, y, test_size, stratify_cols=None, stratify_target=False, random_state=None):
    """
    Split X and y into train and test sets with optional stratification on X columns and/or y target.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Features dataframe
    y : pandas.Series or pandas.DataFrame
        Target variable
    test_size : float
        Proportion of test set (e.g., 0.2 for 20%)
    stratify_cols : list of str, optional
        Column names from X to stratify by
    stratify_target : bool, default False
        Whether to stratify by target variable
    random_state : int, optional
        Random state for reproducibility
        
    Returns
    -------
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame  
        Test features
    y_train : pandas.Series or pandas.DataFrame
        Training target
    y_test : pandas.Series or pandas.DataFrame
        Test target
        
    Notes
    -----
    Prints proportions of stratified columns in train and test sets.
    Uses sklearn.model_selection.train_test_split internally.
    """
    if stratify_cols and stratify_target:
        stratify_data = pd.concat([X[stratify_cols], y], axis=1)
    elif stratify_cols:
        stratify_data = X[stratify_cols]
    elif stratify_target:
        stratify_data = y
    else:
        stratify_data = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_data, random_state=random_state
    )

    cols_to_check = stratify_cols if stratify_cols else []
    if stratify_target and hasattr(y, 'name'):
        cols_to_check = cols_to_check + [y.name if y.name else 'target']
    
    for col in cols_to_check:
        print(f"\n{col} proportions in Train set:")
        print(X_train[col].value_counts(normalize=True).sort_index() if col in X_train else y_train.value_counts(normalize=True).sort_index())
        print(f"\n{col} proportions in Test set:")
        print(X_test[col].value_counts(normalize=True).sort_index() if col in X_test else y_test.value_counts(normalize=True).sort_index())

    return X_train, X_test, y_train, y_test   