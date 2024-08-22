# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 08:08:58 2024

@author: umesh
"""

# Drive mounting
from google.colab import drive

# This will prompt for authorization and mount your Google Drive.
drive.mount('/content/drive')

# Unzipping the file
!gunzip "{file_path}"

# Loading the data set
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Load dataset
file_path = '/content/drive/My Drive/Toys_and_Games_5.json'
df = pd.read_json(file_path, lines=True)

# Handling missing values for summary, reviewer id, and asin id
df['summary'] = df['summary'].fillna('no summary')
df['reviewerID'] = df['reviewerID'].fillna('unknown')
df['asin'] = df['asin'].fillna('unknown')

# Performing text cleaning by removing stop words and converting string to lowercase
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # remove all non-word characters
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if not word in stop_words]
    return ' '.join(filtered_text)

df['clean_summary'] = df['summary'].apply(clean_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['clean_summary'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Encode categorical variables
le = LabelEncoder()
df['reviewerID_encoded'] = le.fit_transform(df['reviewerID'])
df['asin_encoded'] = le.fit_transform(df['asin'])

# Combine all features into one DataFrame
df_final = pd.concat([df[['overall', 'verified']], tfidf_df], axis=1)
df_final['verified'] = df_final['verified'].astype(int)

x = df_final.drop('overall', axis=1)
y = df_final['overall'].astype(float)

# Random forest Regressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error

# Initialize models
rf_model = RandomForestRegressor(n_estimators=1, random_state=42, verbose=1)

# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and calculate RMSE for RandomForest
rf_mse = cross_val_score(rf_model, x, y, cv=kf, scoring='neg_mean_squared_error', verbose=1)
rf_rmse = np.sqrt(-rf_mse)

print("Random Forest RMSE scores for each fold:", rf_rmse)
print("Minimum RF RMSE:", min(np.abs(rf_rmse)))
print("Average RMSE:", np.mean(rf_rmse))

# Linear Regression Model
from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Perform cross-validation and calculate RMSE for Linear Regression
lr_mse = cross_val_score(lr_model, x, y, cv=kf, scoring='neg_mean_squared_error', verbose=1)
lr_rmse = np.sqrt(-lr_mse)

print("Linear regression RMSE scores for each fold:", lr_rmse)
print("Minimum LR RMSE:", min(np.abs(lr_rmse)))
print("Average LR RMSE:", np.mean(lr_rmse))

# XGBoost Model
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

xg_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=5, random_state=42)
param_grid_xg = {
    'n_estimators': [5, 10, 15],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.5, 0.7, 1]
}

grid_search_xg = GridSearchCV(estimator=xg_model, param_grid=param_grid_xg, cv=kf, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search_xg.fit(x, y)

best_xg = grid_search_xg.best_estimator_
xg_rmse = np.sqrt(-cross_val_score(best_xg, x, y, cv=kf, scoring='neg_mean_squared_error', verbose=1))

print("XGBoost RMSE scores for each fold:", xg_rmse)
print("Average XGBoost RMSE:", np.mean(xg_rmse))

# Optimizing RMSE score using GridSearchCV
param_grid = {
    'n_estimators': [10, 20, 30],
    'max_depth': [None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize the RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=4, scoring='neg_mean_squared_error', n_jobs=3, verbose=1)
grid_search.fit(x, y)

best_rf = grid_search.best_estimator_
best_rf_mse = cross_val_score(best_rf, x, y, cv=4, scoring='neg_mean_squared_error')
best_rf_rmse = np.sqrt(-best_rf_mse)

print("Improved Random Forest RMSE scores for each fold:", best_rf_rmse)
print("Improved Average RF RMSE:", np.mean(best_rf_rmse))