# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 12:13:03 2025

@author: 22594
"""

import pandas as pd
import numpy as np
from scipy import stats
from plotnine import *

def categorical_correlation(feature_array, target_array, alpha=0.05,flag_moreinfo=False):
    """
    Compute correlation between a categorical feature and continuous target.
    Parameters:
    -----------
    feature_array : array-like
        Categorical feature values (can be strings, numbers, or mixed)
    target_array : array-like
        Continuous target values (must be numeric)
        
    Returns:
    --------
    float or np.nan
        ANOVA F-statistic (for >2 categories) or point-biserial correlation (binary)
        
    Interpretation Guidelines： 
    Effect Size Benchmarks (Using η² = F/(F + df_error)):
    η² ≥ 0.01: Small effect (weak correlation)    
    η² ≥ 0.06: Medium effect (moderate correlation) 
    η² ≥ 0.14: Large effect (strong correlation)
        
    Examples:
    ---------
    # Nominal  feature example
    feature = np.array(['Peter', 'Jack', 'Jack', 'Ella', 'Peter', 'Jack', np.nan])
    target = np.array([250000, 225000, 190000, 160000, 240000, 195000, 180000])
    corr = categorical_correlation(feature, target)
    
    # Binary feature example  
    feature = np.array(['Y', 'N', 'Y', 'N', 'Y', np.nan])
    target = np.array([210000, 185000, 220000, 175000, 230000, 190000])
    corr = categorical_correlation(feature, target)
    """
    # Convert to pandas Series for robust NA handling
    feature_series = pd.Series(feature_array)
    target_series = pd.Series(target_array)
    
    # Create mask for non-NA values (works for all dtypes)
    mask = feature_series.notna() & target_series.notna()
    feature_clean = feature_series[mask]
    target_clean = target_series[mask]
    
    groups = [target_clean[feature_clean == cat] for cat in feature_clean.unique()]
    
    if all(len(g) > 0 for g in groups):
        f_val, p_val = stats.f_oneway(*groups)
        eta_squared = f_val / (f_val + (len(target_clean) - feature_clean.nunique())) 
        
        if flag_moreinfo:
             return {'f_statistic': f_val,
                    'p_value': p_val,
                    'eta_squared': eta_squared,
                    'significant': p_val < alpha }
        
        else:
            return eta_squared

#============================================================================================================
    
# Load the dataset
df_pd = pd.read_csv("Housing_Prices_Dataset.csv")
# Display the first few rows of the dataframe
print(df_pd.head())

df_pd["SALEPRICE"] = df_pd["SALEPRICE"] / 10000  # Scale SALEPRICE for readability
df_pd["LOTAREA"]=df_pd["LOTAREA"]/10000
df_pd["LOTAREA"]=[ x if x<2 else np.NaN  for x in df_pd["LOTAREA"] ]

columns = df_pd.columns

