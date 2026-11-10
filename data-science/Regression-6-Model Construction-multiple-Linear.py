# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 00:25:28 2025

@author: 22594
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel

# Custom Linear Regression to use absolute coefficients as feature importance
class AbsoluteCoeffLinearRegression(LinearRegression):
    @property
    def feature_importances_(self):
        return np.abs(self.coef_)  # Use absolute coefficients for selection

# Configuration
THRESHOLD = 0.4  # Keep features with |coefficient| >= 1.0 (≈ $10,000 impact)
POLY_DEGREE = 2

# Selected core features
numeric_features = ['GARAGECARS', 'FULLBATH', 'TOTRMSABVGRD', 'YEARBUILT']
nominal_features = ['GARAGEFINISH']
ordinal_features = ['KITCHENQUAL']
# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('nom', OneHotEncoder(handle_unknown='ignore'), nominal_features),
    ('ord', OrdinalEncoder(categories=[['Po', 'Fa', 'TA', 'Gd', 'Ex']]), ordinal_features)])

# Load and prepare data
df = pd.read_csv("Housing_Prices_Dataset.csv")
df["SALEPRICE"] = df["SALEPRICE"] / 10000  # Scale target
df["YEARBUILT"] = 2025 - df["YEARBUILT"]  # Convert to age

# Model pipeline with feature selection
model = Pipeline([
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=POLY_DEGREE, include_bias=False, interaction_only=True)),
    ('feature_selector', SelectFromModel(
            AbsoluteCoeffLinearRegression(),  # Use custom estimator
            threshold=THRESHOLD )),  # Numeric threshold (no string) 
    ('regressor', LinearRegression())])

# Split and train
X = df[numeric_features + nominal_features + ordinal_features]
y = df['SALEPRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):,.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# Get selected features and coefficients
feature_names = model.named_steps['poly'].get_feature_names_out(
    model.named_steps['preprocessor'].get_feature_names_out())
selected_mask = model.named_steps['feature_selector'].get_support()
selected_features = feature_names[selected_mask]
coefs = model.named_steps['regressor'].coef_

print("The number of selected features:", len(coefs))

# Create coefficient report
coef_df = pd.DataFrame({ 'Feature': selected_features,
                        'Coefficient': coefs,
                        'Abs_Impact': np.abs(coefs) * 10000  }).sort_values('Abs_Impact', ascending=False)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
print(f"\nSelected Features (Threshold ≥${THRESHOLD*10000:,.0f} Impact):")
print(coef_df)

