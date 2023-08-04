"""
Parameters that are specified before training a model are called hyperparameters.
Examples: alpha in ridge/lasso regression and k in k-NN.
Successful hyperparameter tuning can make a big difference in model performance.
Once hyperparameter has been set, it is good practice to tune it using cross-validation.
Data still can be split into training and test sets. But, the training set is used for cross-validation.

Approaches for hyperparameter tuning:

- Grid search cross-validation: chose a grid of hyperparameter values and test all of them. This method is great but
the number of models to train can be very large because it is the product of the number of hyperparameters and the
number of values to try for each hyperparameter.

- Randomized search cross-validation: sample a fixed number of hyperparameter values from specified distributions.

"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso, LogisticRegression


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
diabetes_df = pd.read_csv('database/diabetes_clean.csv')

# ----------------------------------------------------------
#                   Grid Search Cross-Validation
# ----------------------------------------------------------

# Select the features and target variable
X1 = diabetes_df.drop('glucose', axis=1).values
y1 = diabetes_df['glucose'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Split the data into folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Time the execution
start_time = time.time()

# Set up the parameter grid
param_grid ={"alpha": np.linspace(0.00001, 1, 20)}

# Instantiate the Lasso regressor
lasso = Lasso()

# Instantiate the GridSearchCV object
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)

# Fit it to the data
lasso_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Lasso Regression Parameters: {}".format(lasso_cv.best_params_))
print("Best score is {}".format(lasso_cv.best_score_))

# Evaluate the model
grid_score = lasso_cv.score(X_test, y_test)
print("Grid search score: ", grid_score)

# Print the execution time
print("Execution time GridSearchCV: ", (time.time() - start_time), "seconds \n \n")

# ----------------------------------------------------------
#                   Randomized Search Cross-Validation
# ----------------------------------------------------------
# Select the features and target variable
X2 = diabetes_df.drop('diabetes', axis=1).values
y2 = diabetes_df['diabetes'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Split the data into folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Time the execution
start_time = time.time()

# Instantiate the LogisticRegression regressor
logreg = LogisticRegression(max_iter=300, solver='liblinear')  # This solver supports both l1 and l2 penalties

# Create the parameter space
params = {"penalty": ["l1", "l2"],
          "tol": np.linspace(0.0001, 1., 50),
          "C": np.linspace(0.1, 1., 50),
          "class_weight": ["balanced", {0: 0.8,
                                        1: 0.2}]}

# Instantiate the RandomizedSearchCV object
logreg_cv = RandomizedSearchCV(logreg, params, cv=kf)

# Fit it to the data
logreg_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))

# Evaluate the model
random_score = logreg_cv.score(X_test, y_test)
print("Randomized search score: ", random_score)

# Print the execution time
print("Execution time RandomizedSearchCV: ", (time.time() - start_time))

# ----------------------------------------------------------
# Note that the random search provides a better score than the grid search in less time. Although,
# also note that different target variables were used in both cases and so, it might not be a fair comparison.


