# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 12:13:03 2025

@author: 22594
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df_pd = pd.read_csv("Housing_Prices_Dataset.csv")
# Display the first few rows of the dataframe
print(df_pd.head())

df_pd["SALEPRICE"] = df_pd["SALEPRICE"] / 10000  # Scale SALEPRICE for readability
df_pd["LOTAREA"]=df_pd["LOTAREA"]/10000
df_pd["LOTAREA"]=[ x if x<2 else np.NaN  for x in df_pd["LOTAREA"]  ]

#=========================================Data Cleaning====================================================
#Handle missing values

# 2. Check for missing values in each column
print("\nMissing values per column:")
print(df_pd.isna().sum())

# 3. Display rows with any missing values
print("\nRows with missing values:")
print(df_pd[df_pd.isna().any(axis=1)])


columns = df_pd.columns

# 4. Visualize missing data


#=============================================Conclusion======================================================================



#=============================================Data tranformation======================================================================



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Selected features
numeric_features = ['GARAGECARS', 'FULLBATH', 'TOTRMSABVGRD', 'YEARBUILT']
nominal_features = ['GARAGEFINISH']
ordinal_features = ['KITCHENQUAL']
target = 'SALEPRICE'

# Ordinal mapping
ordinal_categories = [['Po', 'Fa', 'TA', 'Gd', 'Ex']]  #Quality order

# Create transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('nom', OneHotEncoder(handle_unknown='ignore'), nominal_features),
        ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_features)])

# Create full pipeline
model_transformer = Pipeline([('preprocessor', preprocessor)])

# Load and prepare data
df_pd = pd.read_csv("Housing_Prices_Dataset.csv")

# Data preprocessing
df_pd["SALEPRICE"] = df_pd["SALEPRICE"] / 10000  # Scale target variable
df_pd["LOTAREA"] = df_pd["LOTAREA"].div(10000).where(lambda x: x < 2)  # Cap large values

# Split data
X = df_pd[numeric_features + nominal_features + ordinal_features]
y = df_pd[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform the training and test data
X_train_transformed = model_transformer.fit_transform(X_train)
X_test_transformed = model_transformer.fit_transform(X_test)



