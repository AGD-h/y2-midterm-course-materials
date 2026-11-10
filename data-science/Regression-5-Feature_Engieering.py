# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 00:59:59 2025

@author: 22594
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 00:59:59 2025
@author: 22594
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures

# Selected features
numeric_features = ['GARAGECARS', 'FULLBATH', 'TOTRMSABVGRD', 'YEARBUILT']
nominal_features = ['GARAGEFINISH']
ordinal_features = ['KITCHENQUAL']

preprocessor = ColumnTransformer(transformers=[
                ('num',  StandardScaler(), numeric_features),
                ('nom', OneHotEncoder(handle_unknown='ignore'), nominal_features),
                ('ord',  OrdinalEncoder(categories=[['Po', 'Fa', 'TA', 'Gd', 'Ex']]), ordinal_features)])
#==============================================================================================================
# Load the dataset
df_pd = pd.read_csv("Housing_Prices_Dataset.csv")

# Data preprocessing
df_pd["SALEPRICE"] = df_pd["SALEPRICE"] / 10000  # Scale SALEPRICE
df_pd["YEARBUILT"]=  2025-df_pd["YEARBUILT"]

#Create a pipeline with polynomial features for specific columns
model = make_pipeline(
    preprocessor,
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),  # Adjust as needed
    LinearRegression()
)

# Split data
X = df_pd[numeric_features + nominal_features + ordinal_features]
y = df_pd['SALEPRICE']

# First split: training and temporary (testing + validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):,.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# Get feature names after all transformations====================================================
# 1. Get preprocessor feature names
numeric_names = numeric_features
nominal_names = model.named_steps['columntransformer'].named_transformers_['nom'].get_feature_names_out(nominal_features)
ordinal_names = ordinal_features

# Combine base features
base_features = np.concatenate([numeric_names, nominal_names, ordinal_names])

# 2. Get polynomial interaction names 
all_feature_names = model.named_steps['polynomialfeatures'].get_feature_names_out(input_features=base_features)

print("The number of generated features:", len(all_feature_names))

# 3. Create coefficient DataFrame
coefs = pd.DataFrame({'Feature': all_feature_names,
                    'Coefficient': model.named_steps['linearregression'].coef_
                    }).sort_values('Coefficient', key=abs, ascending=False)

# Formatting
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
print("\nFeature Coefficients (Sorted by Impact):")
print(coefs.reset_index(drop=True))

