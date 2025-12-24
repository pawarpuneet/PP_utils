import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import statsmodels.api as sm
from scipy.stats import pearsonr
import scipy.stats as stats
from pygam import GAM, s, f, LinearGAM
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, DetCurveDisplay, auc, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelBinarizer
from PP_utils import ds_statistics
from typing import Literal
from itertools import cycle

dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; 
colors = cycle([dlblue, dlorange, dlpurple, dldarkred, dlmagenta, 'aqua', 'darkorange', 'cornflowerblue'])
colors_set2 = cycle(['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'])

# The p-value roughly indicates the probability of an uncorrelated system 
# producing datasets that have a Pearson correlation 
# at least as extreme as the one computed from these datasets.
# a test of the null hypothesis that the distributions underlying the samples are 
# uncorrelated and normally distributed
def get_corr_pvalue(arr1,arr2):
    pearson_r, p_value = pearsonr(arr1,arr2)
    return p_value

def plot_normality_check(x,hist_stat='count'):
    _, ax = plt.subplots(1,3,figsize=(15,5), layout='constrained')
    # kernel density plot
    sns.kdeplot(x,ax=ax[0])
    ax[0].set_xlim(np.min(x),np.max(x)) # to match to the range of x
    ax[0].set_title('Kernel Density Plot')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    # scilimits parameter controls the range of exponents for which scientific notation is used; 
    #setting it to (0, 0) ensures that scientific notation is used for all values

    # distribution plot
    sns.histplot(x, kde=False, stat=hist_stat, ax=ax[1]) # other stats: probability, density
    ax[1].set_title('Distribution Plot')

    # Normal Q-Q plot
    sm.qqplot(x,dist=stats.norm, line='q', ax=ax[2])
    ax[2].set_title('Normal Q-Q plot')
    plt.show()
    ds_statistics.normality_tests(x)
 
def plot_multicollinearity_checks(X,corr_matrix_figsize=(12,5)):
    ''' correlation matrix is generated using pd.DataFrame.corr()
    p-value is calculated using pearsonr test from scipy.stats'''
    #Pairplots for X to confirm absence of collinearity
    df = pd.DataFrame(X)
    sns.pairplot(data=df)
    plt.show()
    print('''
          The Pearson correlation coefficient measures the linear relationship between two datasets.
          Test for the statistical significance of the correlation coefficient:
          ρ = population correlation coefficient
          Null hypothesis (H0): ρ = 0 populations are linearly uncorrelated 
          Alternative hypothesis (H1): ρ ≠ 0 populations are linearly correlated 
          p-value < 0.05 => reject H0''')
    # Correlation coefficients to confirm absence of collinearity
    _, ax = plt.subplots(1,2,figsize=corr_matrix_figsize, layout='constrained')
    corr_matrix = df.corr()
    p_value_matrix = df.corr(method=get_corr_pvalue)
    sns.heatmap(corr_matrix, annot=True, ax=ax[0])
    ax[0].set_title("Pearson correlation coefficients")
    # p value of correlation coefficients
    sns.heatmap(p_value_matrix, annot=True, ax=ax[1], cmap=sns.cubehelix_palette(as_cmap=True))
    ax[1].set_title('p-value of correlation coefficients (ignore values on diagonal)')
    plt.show()
    # VIF factor 
    ds_statistics.multicollinearity_VIF(X)

def plot_linearity_checks_for_target(X,y,fit_smooth_curve=True,X_is_categorical=False,y_is_categorical=False,normalize_y_when_X_is_cat=False,normalize_X_when_y_is_cat=False):
    '''check for linearity between dependent and independent vars'''
    if X_is_categorical and y_is_categorical:
        print('''Linearity assumption in regression analysis does not apply to the relationship 
              between two categorical variables.For categorical predictors, the regression model 
              uses dummy variables (indicator functions) to represent group membership. The linearity 
              assumption is inherently satisfied for each individual dummy variable because 
              a straight line always perfectly fits two points—the mean outcome for each of the 
              two categories represented by the dummy variable. ''')
        return
    if not X_is_categorical and not y_is_categorical:
        df = pd.DataFrame(X)
        if y.ndim != 1:
            print('y must be 1 dimensional')
            return
        y = pd.Series(y)

        n = X.shape[1]
        y = y.values
        if fit_smooth_curve:
            print('Smooth line fit is done using P-splines')
        # Plot 4 features in a row
        for i in range(0,n,4):
            # select max four features at a time to plot
            slice_end = i+4 if i+4 < n else None 
            df_sub = df.iloc[:,i:slice_end].copy()
            if not fit_smooth_curve:   
                #Pairplot of Y vs X 
                df_sub['y'] =  y
                sns.pairplot(data = df_sub, x_vars=df_sub.columns[:-1], y_vars='y')
                plt.show()
                
            else:
                
                # Scatterplot of Y vs each feature in X and a smooth line fit
                # create 4 subplots in each row if we have at least 4 features else equal to number of features
                plot_cols = n%4 if i+4>n else 4
                plot_width = plot_cols*2.5
                _ ,ax = plt.subplots(1,plot_cols,figsize=(plot_width,2.5), layout='constrained', sharey=True)
                # create subplot for each feature and add the smooth line 
                for j,column in enumerate(df_sub):
                    x = df_sub.loc[:,column].values
                    # Fit a linear GAM using default params
                    gam = LinearGAM(n_splines=10).fit(x, y)
                    # Using generate_X_grid : array is sorted by feature and uniformly spaced, 
                    # term=0 refers to x, term=1 will be intercept term
                    x_new=gam.generate_X_grid(term=0)
                    y_pred_gam = gam.predict(x_new)
                    if plot_cols!=1:
                        axs = ax[j]
                    else:
                        axs = ax # when there is only single subplot, the syntax doesn't allow subscipt like ax[0]
                    axs.scatter(x,y)
                    axs.plot(x_new,y_pred_gam,color=dlorange,label='smooth line fit')
                    axs.label_outer()
                    axs.set_xlabel(column)   
                if plot_cols!=1:
                    axs = ax[0]
                else:
                    axs = ax            
                axs.set_ylabel('target')
                axs.legend()    
                plt.show()

    elif X_is_categorical:
        df = pd.DataFrame(X)
        df['y'] = y
        df['y_norm'] = stats.zscore(df['y'])
        if normalize_y_when_X_is_cat:
            target = 'y_norm'
        else:
            target = 'y'

        for col in df.iloc[:,:-2]:
            _,ax = plt.subplots(1,3,figsize=(15,5),layout='constrained')
            sns.boxplot(data=df,x=col,y=target,hue=col,ax=ax[0])
            sns.kdeplot(data=df,x=target,hue=col,fill=True,ax=ax[1])
            sns.ecdfplot(data=df,x=target,hue=col)
            ax[0].set_title(f'Box-plot of target vs explanatory var:{col}')
            ax[1].set_title(f'KDE plot of target vs explanatory var:{col}')
            ax[2].set_title(f'Empirical cdf plot of target vs explanatory var:{col}')
            plt.show()
       
    elif y_is_categorical:
        df = pd.DataFrame(X)
        if normalize_X_when_y_is_cat:
            df = (df-df.mean())
        df['y'] = y
        target = 'y'
        for col in df.iloc[:,:-1]:
            _,ax = plt.subplots(1,3,figsize=(15,5),layout='constrained')
            sns.boxplot(data=df,x=target,y=col,hue=target,ax=ax[0])
            sns.kdeplot(data=df,x=col,hue=target,fill=True,ax=ax[1])
            sns.ecdfplot(data=df,x=col,hue=target)
            ax[0].set_title(f'Box-plot of target vs explanatory var:{col}')
            ax[1].set_title(f'KDE plot of target vs explanatory var:{col}')
            ax[2].set_title(f'Empirical cdf plot of target vs explanatory var:{col}')
            plt.show()        

# def plot_linearity_checks_for_target_vs_cat(X,y,normalize_target=True):
    
#     df = pd.DataFrame(X)
#     df['y'] = y
#     df['y_norm'] = stats.zscore(df['y'])
#     if normalize_target:
#         target = 'y_norm'
#     else:
#         target = 'y'

#     for col in df.iloc[:,:-2]:
#         _,ax = plt.subplots(1,3,figsize=(15,5),layout='constrained')
#         sns.boxplot(data=df,x=col,y=target,hue=col,ax=ax[0])
#         sns.kdeplot(data=df,x=target,hue=col,fill=True,ax=ax[1])
#         sns.ecdfplot(data=df,x=target,hue=col)
#         ax[0].set_title(f'Box-plot of target vs explanatory var:{col}')
#         ax[1].set_title(f'KDE plot of target vs explanatory var:{col}')
#         ax[2].set_title(f'Empirical cdf plot of target vs explanatory var:{col}')
#         plt.show()

      



# define a function to plot 3 plots for predicted vs actual
# a distribution plot, a scatter plot and a residual plot
def plot_actual_vs_predicted(y,y_pred,X=None,figsize=(15,5),data_label='', run_statistical_tests=False):
    '''X is only used for statistical tests, plotting is all done using y,y_pred only'''
    print(f'{"R-squared:":20} {r2_score(y,y_pred):.4f}')
    print(f'{"Mean Squared Error:":20} {mean_squared_error(y,y_pred):.2e}')
    print(f'{"Mean Absolute Error:":20} {mean_absolute_error(y,y_pred):.2e}')
    ######### Set of plot for actual vs predicted #########
    _, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=figsize, layout='constrained')
    # distribution plot
    # For the KDE in histplot to match kdeplot, use stat="density", 
    # ensuring both the histogram and KDE are on the same density scale.
    # stat="probability": The heights of the bars sum to 1.
    # stat="density": The area of all bars sums to 1
    sns.histplot(y,fill=True, stat='count', kde=False, label='Actual', ax=ax1)
    sns.histplot(y_pred, fill=True, stat='count', kde=False, label='Predicted',ax=ax1)
    ax1.legend()
    ax1.set_title(f'Distribution Actual vs Predicted {data_label}')
    # scatter plot with y=x line
    ax2.scatter(y_pred,y, c='g')
    lims = [
        np.min([ax2.get_xlim(), ax2.get_ylim()]),
        np.max([ax2.get_xlim(), ax2.get_ylim()])
    ]
    ax2.plot(lims,lims, c="#C8CAC8", linestyle='--')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title(f'Actual vs Predicted {data_label}')

    # kernel density plot of predictions vs actual values
    sns.kdeplot(y, color=dlblue, ax = ax3,label='Actual')
    sns.kdeplot(y_pred, color=dlorange, ax= ax3, label='Predicted')
    ax3.set_xlim(ax1.get_xlim()) # to match the range of x axis to the distribution plot
    ax3.set_title(f'Kernel Density Plot Actual vs Predicted {data_label}')
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    # scilimits parameter controls the range of exponents for which scientific notation is used; 
    #setting it to (0, 0) ensures that scientific notation is used for all values

    ######### Set of plot for residuals #########
    fig, ax = plt.subplots(1,3,figsize =(15,5), layout='constrained')
    residuals = y - y_pred 
    # residuals distribution for normality check
    sns.histplot(residuals, kde=True, ax=ax[0])
    ax[0].set_xlabel('Residuals (actual-predicted)')
    ax[0].set_title(f'Distribution of residuals {data_label}')

    # residual plot for homoscedasticity check (check for constant variance)
    ax[1].scatter(y_pred, residuals, marker='o', c='b' )
    ax[1].axhline(y=0, c="#C8CAC8", linestyle='--')
    ax[1].set_xlim(lims)
    ax[1].set_title(f'Residuals vs. Predicted {data_label}')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('Residuals (actual-predicted)')

    # Normal Q-Q plot for residuals
    sm.qqplot(residuals, line='q', ax=ax[2])
    ax[2].set_title(f'Normal Q-Q plot of residuals {data_label}')
 
    plt.show()
    if run_statistical_tests:
        # White's test for homoscedasticity
        print('Tests for Homoscedasticity of residuals:')
        print('----------------------------------------')
        ds_statistics.homoscedasticity_tests(X,y,y_pred)
        print('Tests for normality of residuals:')
        print('---------------------------------')
        ds_statistics.normality_tests(residuals)

def plot_prediction_by_feature(X,y,y_pred,features=[],data_label=''):
    if X.ndim == 1:
        n_features = 1
    else:
        n_features = X.shape[1]
    if len(features) != n_features:
        print("length of features list does not match the number of columns in X")
        return
    features = features
    df = pd.DataFrame(X)
    df['y'] = y
    df['y_pred'] = y_pred
    df.columns = features + [f'y {data_label}', f'y_pred {data_label}']
    df_long = pd.melt(df,id_vars=df.columns[0:-2], value_vars=df.columns[-2:], value_name='target_value', var_name='label')
    for i in range(0,n_features,4):
        slice_end = i+4 if i+4 < n_features else None
        sns.pairplot(df_long, x_vars=features[i:slice_end], y_vars='target_value', kind='scatter',hue='label')
    plt.show()

def plot_poly_reg_1d(x,y,degree=2,feature_name='x',target_name='y',num_points_x=100, data_label='', x_range=[]):
    if x.ndim > 1:
        print('x must be 1 dimensional.')
        return
    coef = np.polyfit(x, y, degree)
    model = np.poly1d(coef) # model
    if x_range:
        x_min = x_range[0]
        x_max = x_range[1]
    else:
        x_min = np.min(x)
        x_max = np.max(x)
    x_new = np.linspace(x_min,x_max,num_points_x)
    print(f'''Prediction is done using {num_points_x} values of {feature_name} 
    in the range between {x_min} and {x_max} both inclusive. 
    These values are generated using np.linspace({x_min},{x_max},{num_points_x})''')
    y_pred = model(x_new)
    # x_sorted = x.sort_values()
    # y_pred = model(x_sorted)
    plt.plot(x, y, '.', label = 'Actual')
    plt.plot(x_new, y_pred, '-', label = 'Predicted')
    plt.title(f'Polynomial Fit: {target_name} vs {feature_name} {data_label}')
    plt.legend()
    plt.xlabel(f'{feature_name}')
    plt.ylabel(f'{target_name}')
    plt.show()
    return model

def plot_GridSearchCV_scores_by_split(cv_results_df, scorer_name='score',n_splits = 30,ranks_list=None):
    '''function to plot GridSearch score by CV split for all given models
    This shows the dependency between cv fold and the score
    Since some partitions of the data can make the model fitting particularly easy 
    or hard for all models, the models scores will co-vary.
    This plot can also be used as a visual tool for comparing different model.'''
    # Following steps are done to get cv_results_df:
    # df = pd.DataFrame(grid_search_model.cv_results_)
    # if multi_metric_eval and scorer_name == 'score':
    #     print('Must provide a scorer name in case of multi metric evaluation')
    #     return
    # sort_by_col = 'rank_test_' + scorer_name
    # df = df.sort_values(by=[sort_by_col])
    # df = df.set_index(
    #     df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    #     ).rename_axis("param_comb")
        
    # filter columns to get score for each CV split
    model_scores = cv_results_df.filter(regex=f"split\d*_test_{scorer_name}")
    # iloc doesn't raise out-of-bounds error when using slicing, so let's put a check on n_splits
    max_splits = len(model_scores.columns)
    if n_splits > max_splits:
        print(f'There are only {max_splits} CV splits total')
        n_splits = max_splits
    # Let's plot test score cv fold 
    fig, ax = plt.subplots()
    if ranks_list:
        ranks_list = [n-1 for n in ranks_list]
        sns.lineplot(
            data = model_scores.transpose().iloc[:n_splits,ranks_list], # default: plot for 30 splits and given param sets
            dashes=False,
            palette='Set2',
            ax=ax
        )
    else:
        sns.lineplot(
            data = model_scores.transpose().iloc[:n_splits,:], # default: plot for 30 splits and all param sets
            dashes=False,
            palette='Set2',
            ax=ax
        )        
    plt.tick_params(bottom=True, labelbottom=False)
    plt.xlabel(f'CV test splits 0-{n_splits}')
    plt.ylabel(f'{scorer_name}')
    plt.show()

def clf_confusion_matrix(y,y_pred,classes=[],title_label='',print_classification_report=True):
    '''Calculates confusion matrix and plots it.
    Also print classification report'''
    cm = confusion_matrix(y,y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix {title_label}', fontsize=15, pad=20)
    ax = plt.gca()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    plt.show()
    classes_str = [str(label) for label in classes]
    if print_classification_report:
        print(classification_report(y, y_pred, target_names=classes_str))

def clf_confusion_matrix_from_estimator(X,y,classifiers={}):
    '''Calculates confusion matrix and plots it.
    Also print classification report'''
    clfs = [(name,clf) for name,clf in classifiers.items()]

    for i in range(0,len(clfs),2):
        _, ax = plt.subplots(1,2,figsize=(10,5), layout='constrained')
        ConfusionMatrixDisplay.from_estimator(clfs[i][1],X,y,display_labels=clfs[i][1].classes_,cmap=plt.cm.Blues,ax=ax[0])
        ax[0].set_title(f'Confusion Matrix {clfs[i][0]}')
        ax[0].xaxis.set_label_position('top')
        ax[0].xaxis.tick_top()
        if i+1 < len(clfs):
            ConfusionMatrixDisplay.from_estimator(clfs[i+1][1],X,y,display_labels=clfs[i+1][1].classes_,cmap=plt.cm.Blues,ax=ax[1])
            ax[1].set_title(f'Confusion Matrix {clfs[i+1][0]}')
            ax[1].xaxis.set_label_position('top')
            ax[1].xaxis.tick_top()
        plt.show()
    

def clf_roccurve_comparison(X, y, estimators=[]):
    '''Uses RocCurveDisplay.from_estimator and plots roc curves
    for a list of estimators for comparison'''
    est0_disp = RocCurveDisplay.from_estimator(estimators[0], X, y)
    for est in estimators[1:]:
        RocCurveDisplay.from_estimator(est, X, y, ax=est0_disp.ax_)
    est0_disp.figure_.suptitle("ROC curve comparison")
    plt.show()

def clf_classification_curves(X, y, classifiers={}, add_dummy_classifier=False,fit_classifiers = False, X_train=None, y_train=None):
    '''
    Uses:
    RocCurveDisplay.from_estimator
    PrecisionRecallDisplay.from_estimator
    DetCurveDisplay.from_estimator
    
    Parameters
    ----------
    X : matrix like (test set)
    y : target (test set)
    classifiers : dict containing
        { "name1": classifier1,
          "name2": classifier2,
          ...
        }
    add_dummy_classifier: the baseline to define the chance level 
        is a classifier that would always predict the most frequent class.
    fit_classifiers: to fit training data to estimators
    X_train : optional , needed only if add_dummy_classifier and/or fit_classifiers is True
    y_train: optional , needed only if add_dummy_classifier and/or fit_classifiers is True

    Returns
    -------
    None
    '''
    if add_dummy_classifier or fit_classifiers:
        if X_train is None or  y_train is None:
            print("X_train and y_train are required if add_dummy_classifier and/or fit_classifiers is True")
            return
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    
    for (name, clf),color in zip(classifiers.items(),colors_set2):
        if fit_classifiers:
            clf.fit(X_train,y_train)
        RocCurveDisplay.from_estimator(clf,X,y,name=name,ax=ax[0],curve_kwargs = dict(color=color))
        PrecisionRecallDisplay.from_estimator(clf,X,y,name=name,ax=ax[1],color=color)
        DetCurveDisplay.from_estimator(clf,X,y,name=name,ax=ax[2],color=color)
    if add_dummy_classifier:
        dummy = DummyClassifier()
        dummy.fit(X_train,y_train)
        RocCurveDisplay.from_estimator(dummy,X,y,name="Non-informative baseline",ax=ax[0],curve_kwargs = dict(color="#C8CAC8",linestyle='--'))
        PrecisionRecallDisplay.from_estimator(dummy,X,y,name="Non-informative baseline",ax=ax[1],color="#C8CAC8",linestyle='--')
        DetCurveDisplay.from_estimator(dummy,X,y,name="Non-informative baseline",ax=ax[2],color="#C8CAC8",linestyle='--')
    ax[0].set_title('ROC curve')
    ax[1].set_title('Precision-Recall curve')
    ax[2].set_title("Detection Error Tradeoff (DET) curve")
    plt.legend()
    plt.show()
    print('''DummyClassifier is a classifier that makes constant predictions, 
    independent of the input features in X, making it a non-informative classifier.''')

def clf_roc_curve_with_cross_val(classifier,X,y,n_splits=6):
    '''Cross validate roc-curve using StratifiedKFold with 6 splits
    and plot the variation'''
    n_splits = n_splits
    cv = StratifiedKFold(n_splits=n_splits)
    cv_results = cross_validate(
        classifier, X, y, cv=cv, return_estimator=True, return_indices=True
    )
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    curve_kwargs_list = [
        dict(alpha=0.3, lw=1, color=colors[fold % len(colors)]) for fold in range(n_splits)
    ]
    names = [f"ROC fold {idx}" for idx in range(n_splits)]

    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []

    _, ax = plt.subplots(figsize=(6, 6))
    viz = RocCurveDisplay.from_cv_results(
        cv_results,
        X,
        y,
        ax=ax,
        name=names,
        curve_kwargs=curve_kwargs_list,
        plot_chance_level=True,
    )

    for idx in range(n_splits):
        interp_tpr = np.interp(mean_fpr, viz.fpr[idx], viz.tpr[idx])
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(viz.roc_auc)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(interp_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability\n(Positive label '{classifier.classes_[1]}')",
    )
    ax.legend(loc="lower right")
    plt.show()    

################# Multiclass classification #################
#############################################################
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py  #

def clf_roc_curve_one_vs_rest_for_multiclass(X_train, y_train, X_test, y_test, classifier, class_of_interest):
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    #y_onehot_test.shape  # (n_samples, n_classes)
    class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
    display = RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_score[:, class_id],
        name=f"{class_of_interest} vs the rest",
        curve_kwargs=dict(color="darkorange"),
        plot_chance_level=True,
        despine=True,
    )
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"One-vs-Rest ROC curves:\n{class_of_interest} vs the rest",
    )    

def clf_roc_curve_OvR_micro_avg_for_multiclass(X_train,y_train,X_test,y_test,classifier):
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)    
    display = RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_score.ravel(),
        name="micro-average OvR",
        curve_kwargs=dict(color="darkorange"),
        plot_chance_level=True,
        despine=True,
    )
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Micro-averaged One-vs-Rest\nReceiver Operating Characteristic",
    )

def clf_roc_curve_OvR_macro_avg_for_multiclass(X_train,y_train,X_test,y_test,classifier,n_classes):
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)        
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    #print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")    
    plt.plot(
    fpr["macro"],
    tpr["macro"],
    label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
    color="navy",
    linestyle=":",
    linewidth=4,
    )
    ax = plt.gca()
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Macro-averaged One-vs-Rest\nReceiver Operating Characteristic",
        xlim=(0.0,1.0),
        ylim=(0.0,1.0)
    )
    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)

def clf_roc_curve_OvR_for_multiclass(X_train,y_train,X_test,y_test,classifier,n_classes,classes_names):
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)       
    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")    

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = colors_set2 #cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {classes_names[class_id]} vs the rest",
            curve_kwargs=dict(color=color),
            ax=ax,
            plot_chance_level=(class_id == 2),
            despine=True,
        )

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Receiver Operating Characteristic\nfor One-vs-Rest multiclass",
    )    