"""
This model is used for classification problems.
It is similar to linear regression, but the output is a probability between 0 and 1.
The output is then converted to a binary value of 0 or 1.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             classification_report)
import matplotlib.pyplot as plt

# Load the data
diabetes_df = pd.read_csv('database/diabetes_clean.csv')

# Select the features and target variable
X = diabetes_df.drop('diabetes', axis=1).values
y = diabetes_df['diabetes'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the classifier
# Default iterations were set to 100, but it is not enough for this dataset (warning raised)
logreg = LogisticRegression(max_iter=250)

# Fit the classifier
logreg.fit(X_train, y_train)

# Predict the labels
y_pred = logreg.predict(X_test)

# Predict probabilities
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# Print the probabilities for the first 10 observations
print(y_pred_prob[:10])

# Print the y_pred for the first 10 observations
print(y_pred[:10])

# Let's analyze how the true positive rate and false positive rate change as the threshold changes
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)  #fpr: false positive rate, tpr: true positive rate

# An ROC curve plots the true positive rate versus the false positive rate
# for every possible classification threshold. Lowering the classification threshold
# classifies more items as positive, thus increasing both the false positive rate and
# true positive rate.

# Create the plot
fig, ax= plt.subplots()
ax.plot([0, 1], [0, 1], 'k--')
ax.plot(fpr, tpr, label='Logistic Regression')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Logistic Regression ROC Curve')
ax.legend()
plt.show()

# Analysis of the ROC curve
# The closer the curve is to the top-left corner, the better the model,
# because it means that the true positive rate is high and the false positive rate is low.
# It can be seen that the ROC curve is above the diagonal, which is a good sign.

# Compute the area under the ROC curve
print('AUC: {} \n'.format(roc_auc_score(y_test, y_pred_prob)))
# AUC provides an aggregate measure of performance across all possible classification
# thresholds. AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong
# has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.
# In summary, AUC tells you how much the model is capable of distinguishing
# between classes.

# Compute the confusion matrix
print('\n Confusion matrix')
print(confusion_matrix(y_test, y_pred))

# Compute the classification report
print('\n Classification report')
print(classification_report(y_test, y_pred))

# Note: It is interesting to note that the logistic regression performs better than the
# k-NN classifier, even though the k-NN classifier (see the metrics from script 6_classification.py)





