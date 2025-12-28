""" Data preprocessing related utility functions"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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

    