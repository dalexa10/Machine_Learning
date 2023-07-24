import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

# Load the data
diabetes_df = pd.read_csv('database/diabetes_clean.csv')
# print(diabetes_df.head())
print(diabetes_df.columns)

# Clean data by dropping the rows with zero values
diabetes_df = diabetes_df[diabetes_df['glucose'] != 0]
diabetes_df = diabetes_df[diabetes_df['bmi'] != 0]

# Select the features
X = diabetes_df.drop('glucose', axis=1).values
y = diabetes_df['glucose'].values

# Fold the data into 6 folds
kf = KFold(n_splits=6, shuffle=True, random_state=42)
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=kf)

# Compute the mean squared error for each fold
print('Mean squared error for each fold: {}'.format(cv_results))

# Compute the mean squared error
mse = cv_results.mean()
std = cv_results.std()
print('Mean squared error {:.5f} and std dev {:.5f}'.format(mse, std))

# Compute confidence interval
print('95% confidence interval: {}'.format(np.quantile((cv_results), [0.025, 0.975])))

