"""
Accuracy is not usually a good metric for classification problems,
especially when you have class imbalance (uneven frequency of classes).
Hence, it needs a difference approach to assess the performance of a model.

The goal is to predict weather or not each individual has diabetes based on
features such as age and body mass index (BMI). Thus, it is a binary classification
problem. A value of 0 indicates that the individual does not have diabetes, while
a value of 1 indicates that the individual has diabetes.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
diabetes_df = pd.read_csv('database/diabetes_clean.csv')
# print(diabetes_df.head())
print(diabetes_df.columns)

# Clean data by dropping the rows with zero values
diabetes_df = diabetes_df[diabetes_df['glucose'] != 0]
diabetes_df = diabetes_df[diabetes_df['bmi'] != 0]
diabetes_df = diabetes_df[diabetes_df['age'] != 0]

# Select the features
X = diabetes_df[['bmi', 'age']].values
y = diabetes_df['diabetes'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the classifier
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = knn.predict(X_test)

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))

# The classification matrix has the following shape:

# [[TN, FP],
#  [FN, TP]]
# where:
# TN: True Negative: Actual value is negative and predicted value is negative
# FP: False Positive: Actual value is negative and predicted value is positive
# FN: False Negative: Actual value is positive and predicted value is negative
# TP: True Positive: Actual value is positive and predicted value is positive

# Print the classification report
print(classification_report(y_test, y_pred))
# Note that the output shows a better F1 score for the zero class, which represents
# individuals that do not have diabetes. This is because there are more individuals

# In the report printed above,
# - The precision is the ratio of the true positives to the sum of true and false positives. (Positive predictive value)
# - The recall is the ratio of the true positives to the sum of true positives and false negatives. (Sensitivity)
# - The F1 score is the harmonic mean of the precision and recall. (F1 score)

# The metric that is most important depends on the problem and the business goal.

