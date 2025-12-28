# Data Cleaning Utilities
# This module provides utility functions for data cleaning tasks.
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


""" The function below uses seaborn.heatmap() to represent the correlation
between columns with missing values. The values are represented as a
heatmap, where the color intensity indicates the strength of the correlation.
This is useful for identifying relationships between missing values in different columns. 
Values very close to 0, where there is little to no relationship, aren't labeled.
Values close to -1 indicate a near perfect negative correlation. 
That means for almost every row that has a null value in one column, 
the other has a non-null value and vice-versa.
"""
def plot_null_correlations(df):

    # create a correlation matrix only for columns with at least
    # one missing value
    cols_with_missing_vals = df.columns[df.isnull().sum() > 0]
    missing_corr = df[cols_with_missing_vals].isnull().corr()
    
    # create a mask to avoid repeated values and make
    # the plot easier to read
    missing_corr = missing_corr.iloc[1:, :-1]
    mask = np.triu(np.ones_like(missing_corr), k=1)
    
    # plot a heatmap of the values
    plt.figure(figsize=(20,14))
    ax = sns.heatmap(missing_corr, vmin=-1, vmax=1, cbar=False,
                     cmap='RdBu', mask=mask, annot=True)
    
    # format the text in the plot to make it easier to read
    for text in ax.texts:
        t = float(text.get_text())
        if -0.05 < t < 0.01:
            text.set_text('')
        else:
            text.set_text(round(t, 2))
        text.set_fontsize('x-large')
    plt.xticks(rotation=90, size='x-large')
    plt.yticks(rotation=0, size='x-large')

    plt.show()

""" The function below uses seaborn.heatmap() to represent null values 
as light squares and non-null values as dark squares.
This is useful for visualizing the distribution of missing values
across the DataFrame. 
Sorting the DataFrame is recommended before using this function.
This will gather some of the null and non-null values together and make patterns more obvious."""
def plot_null_matrix(df, figsize=(18,15)):
    # initiate the figure
    plt.figure(figsize=figsize)
    # create a boolean dataframe based on whether values are null
    df_null = df.isnull()
    # create a heatmap of the boolean dataframe
    sns.heatmap(~df_null, cbar=False, yticklabels=False)
    plt.xticks(rotation=90, size='x-large')
    plt.show()

def plot_neat_histogram(data, n_bins, range=None, title='', grid=True):
    """ Creates a histogram of the given series with x ticks aligned with the bins

    Parameters
    ----------
    data : pd.Series of shape (n_samples,)
    range : specifies the lower and upper bounds of the bins for the histogram
            the bins will be constructed within this specified range
            Any data points outside this range are ignored during the histogram computation
    grid: grid=True adds a grid to the plot for better readability
    Returns
    -------
    None
    """
    if range:
        start = range[0]
        end = range[1]
    else:
        start = data.min()
        end = data.max()
    step = (end-start)/n_bins
    xticks = np.arange(start,end+0.1,step)
    data.plot.hist(range=range,bins=n_bins,xticks=xticks,grid=True)
    plt.title(title)
    plt.show()

def plot_categorical_bars(df, columns, rows=2, cols=3, show_percent=True):
    """
    Plot bar charts for categorical columns with optional percentage or count labels.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    columns : list of str
        List of column names to plot.
    rows : int, optional
        Number of rows in the subplot grid, by default 2.
    cols : int, optional
        Number of columns in the subplot grid, by default 3.
    show_percent : bool, optional
        If True, display percentages; if False, display counts, by default True.
    
    Returns
    -------
    None
        Displays the plot using matplotlib.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), sharey=True)
    axes = axes.ravel()
    
    for i, col in enumerate(columns):
        counts = df[col].value_counts()
        if show_percent:
            data = counts / len(df) * 100
            labels = [f'{v:.1f}%' for v in data]
        else:
            data = counts
            labels = [str(v) for v in data]
            
        ax = data.plot(kind='bar', ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_yticks([])
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xlabel('') # column name is already in the title
        
        for container in ax.containers:
            ax.bar_label(container, labels=labels, fontsize=8)
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()   


def plot_categorical_bars_grouped_by(df, categorical_cols, group_by_col='target', rows=2, cols=3, show_percent=True,percent_out_of='all'):
    """
    Plot grouped bar charts for categorical columns, normalized by category, with a single legend.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    categorical_cols : list of str
        List of categorical column names to plot.
    group_by_col : str, optional
        Column name to group by, by default 'target'.
    rows : int, optional
        Number of subplot rows, by default 2.
    cols : int, optional
        Number of subplot columns, by default 3.
    show_percent : bool, optional
        If True, display percentages; if False, display counts, by default True.
    percent_out_of: 'all' or 'columns' or 'index'
    Returns
    -------
    None
        Displays the plot.
    """
    # Create subplots with shared y-axis
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), sharey=True)
    axes = axes.ravel()  # Flatten axes array for easy indexing

    for i, col in enumerate(categorical_cols):
        # Compute crosstab (frequency or percentage based on show_percent)
        ctab = pd.crosstab(df[col], df[group_by_col])
        if show_percent:
            ctab = pd.crosstab(df[col], df[group_by_col], normalize=percent_out_of) * 100   # Normalize based on all observations, 
            #other options are 'column' or 'index'
        ctab.plot(kind='bar', ax=axes[i], legend=False)  # Plot without individual legends
        axes[i].set_title(col)           # Set title to column name
        axes[i].set_xlabel('')           # Remove x-label to avoid redundancy
        axes[i].set_yticks([])           # Hide y-axis ticks
        axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
        # Add formatted labels (percentage or count) on bars
        for container in axes[i].containers:
            fmt = '%.1f%%' if show_percent else '%d'
            axes[i].bar_label(container, fmt=fmt, fontsize=8)

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    # Extract legend handles and labels from the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    # Add a single, centralized legend for the entire figure
    fig.legend(handles, labels, title=group_by_col, loc='upper center', bbox_to_anchor=(0.5, 0.95))

    # Adjust layout to make room for the legend and prevent overlap
    plt.tight_layout()
    plt.show()   

