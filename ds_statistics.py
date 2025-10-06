from scipy.stats import (kstest, shapiro, normaltest, jarque_bera, 
                         cramervonmises, anderson)
from scipy.stats import (norm, uniform, triang, expon, arcsine, gamma, skew, skewtest, kurtosis)
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.compat import lzip
from sklearn.preprocessing import StandardScaler
from collections import namedtuple
from tabulate import tabulate
import numpy as np
import pandas as pd

def pass_fail(test):
    '''make decisions whether pass or fail with a significance level of 5 %'''

    result = ["don't reject H0" if test[i].pvalue >= 0.05 else 'reject H0' for i in range(6)]
    result.append("don't reject H0" if test[6].critical_values[2]>=test[6].statistic 
                  else 'reject H0')
    return result

def print_out(out, result):
    '''print out the test result using tubulate library
       out: from normality_tests function
       result: from pass_fail function'''
    
    # method names from function.__doc__
    method = [normality_tests.__doc__.translate(
             str.maketrans('', '', '\n')).split(",")[i].
             strip() for i in range(1,8)]
    array =[]
    for i in range(len(out)-1):                      # from method 1 to 7
        sub_array = [f'{method[i]:20}', f'{out[i].statistic:8.4f}', 
                     f'{out[i].pvalue:8.4f}', result[i]]
        array.append(sub_array)  
    # a significance level of 5 % for method 8    
    array.append([method[-1], f'{out[-1].statistic:.4f}', 
                  f'{out[-1].critical_values[2]:.4f}*', result[-1]])
    print(tabulate(array, headers=["test methods", "statistic", 
                                "p-value*", "result"]))
    print('*Not a p-value but a critical-value for a signif.level of 5%')
    print('''
          'result' is based on significance level of 0.05
          Null (H0): Data is normally distributed.
          Alternative (H1): Data is not normally distributed
          ''')
def normality_tests(samples):
    ''' normality tests using all available python libraries
        number_of_bins: effective only for chi-square test,
        1.Kolmogorov-Smirnov test,
        2.Shapiro-Wilk test,
        3.D'Agostino's K-squared test,
        4.Jarque–Bera test,
        5.Lilliefors test,
        6.Cramér–von Mises criterion,
        7.Anderson-Darling test
        '''
    samples = pd.Series(samples)
    # Standardize
    mean = np.mean(samples)
    stdev = np.std(samples)
    samples = (samples-mean)/stdev

    # Lilliefors renamed using named tuple to hamornize its name with others
    LillieforsResult = namedtuple('LillieforsResult', ['statistic', 'pvalue'])
    lil = lilliefors(samples)
    
    # for chi-square test
    # f_obs, bins =  np.histogram(samples, bins=number_of_bins_for_chi_square)
    # bin_probability = [norm.cdf(bins[i+1]) - norm.cdf(bins[i])
    #          for i in range(len(bins)-1)]
    # f_exp = np.array(bin_probability)*len(samples)/np.sum(bin_probability)       # normalized
    # ChisquareResult = namedtuple('ChisquareResult',['statistic', 'pvalue'])
    # chi = chisquare(f_obs=f_obs, f_exp=f_exp)
    
    metrics= [kstest(samples, 'norm'),          
            shapiro(samples),                 
            normaltest(samples),   
            jarque_bera(samples), 
            LillieforsResult(lil[0], lil[1]), 
            cramervonmises(samples, 'norm'),
            #ChisquareResult(chi[0], chi[1]),
            anderson(samples)]
    result = pass_fail(metrics)
    print_out(metrics,result)
    skew_val = skew(samples,bias=True)
    z_statistic, p_val = skewtest(samples)
    kurt_val = kurtosis(samples, fisher=True, bias=True)
    print(f'''
    Skewness = {round(skew_val,4)}, statistic = {round(z_statistic,4)}, p_value = {round(p_val,4)}
    ----------------------------------------------------------------
    Null (H0) : skewness of the population that the samples are drawn from is same as that of a  normal distribution.
    using Fisher-Pearson coefficient of skewness:
    values outside the range of -1 to +1 suggest significant skewness,
    skewness = 0 distribution is symmetrical, skewness > 0 right skew, skewness < 0 left skew

    Kurtosis = {round(kurt_val,4)}
    -------------------------------
    using Fisher's definition for kurtosis:
    kurtosis = 0 => normal distribution
    kurtosis > 0 => leptokurtic (heavier tails i.e. more outliers and sharper peak)
    kurtosis < 0 => platykurtic (lighter tails i.e. fewer outliers and a flatter peak)''')

def homoscedasticity_tests(X,y,y_pred,significance_level = 0.05):
    '''White’s Lagrange Multiplier Test has the null hypothesis that the errors are have same variance or homoscedastic. 
    Having a p-value ≤ 0.05 would indicate that the null hypothesis is rejected, hence Heteroscedasticity.
    White’s test is an asymptotic test, which is meant to be used on large samples'''
    residuals = y - y_pred
    X = pd.DataFrame(X)
    y_pred = pd.Series(y_pred)
    #add constant to predictor variables, required for this test for counting dof as per statsmodels explanation
    X = sm.add_constant(X)
    # White’s Lagrange Multiplier Test for Heteroscedasticity.
    lm_statistic,lm_pvalue,f_statistic,f_pvalue = het_white(residuals,  X)
    lm_result = 'reject H0' if lm_pvalue <significance_level else "don't reject H0"
    f_result = 'reject H0' if f_pvalue <significance_level else "don't reject H0"
    # tabulating results
    arr=[]
    arr.append(["White’s_Lagrange_Multiplier_Test", f'{lm_statistic:8.4f}', 
                    f'{lm_pvalue:8.4f}', lm_result])
    arr.append(['F-Test', f'{f_statistic:8.4f}', 
                    f'{f_pvalue:8.4f}', f_result])
    print(tabulate(arr, headers=["test methods", "statistic", 
                            "p-value", "result"]))
    print(f'''\n
    'result' is based on significance level of {significance_level} 
    Given X = explanatory variables
    Null (H0): Homoscedasticity is present (residuals are equally scattered, and its' variance doesn't depend on X)
    Alternative (H1): Heteroscedasticity is present (residuals are not equally scattered, it's variance depends on X)
              ''')

def multicollinearity_VIF(X):
    '''Variance Inflation Factor (VIF): calculate the VIF for each independent variable. 
    A VIF value greater than 5 or 10 indicates significant multicollinearity.
    using statsmodels.stats.outliers_influence.variance_inflation_factor
    It can be interpreted as :
    1= Not correlated
    1–5 = Moderately correlated
    >5 = Highly correlated
    '''
    X = pd.DataFrame(X)
    if X.shape[1]==1:
        print('X must be multidimensional.')
        return
    # create dataframe for saving VIF values for each column 
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif["feature"] = X.columns
    def result(vif_value):
        if vif_value ==1:
            return 'Not correlated'
        elif vif_value > 1 and vif_value <=5:
            return 'Moderately correlated'
        elif vif_value > 5:
            return 'Highly correlated'
        return 'something went wrong'
    vif['result'] = vif['VIF'].apply(result)
    print('''
    Each independent variable is tested against the rest to calclulate VIF(Variance Inflation Factor).
    VIF for each independent variable:      ''')
    print("-------------------------------------------------")
    print(vif.round(1))
    print('''
    Here is how it can be interpreted :
    1= Not correlated
    1–5 = Moderately correlated
    >5 = Highly correlated      
''')