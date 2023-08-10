"""
This is a similar example as the one in 9_Supervised_Learning/09_classification_models.py,
but here classification models are compared. For that aim, the popularity column in a music dataset
was converted to binary values (0 and 1) and the models were evaluated using the accuracy score.
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error


# Import data
music_df = pd.read_csv('database/music_clean.csv')

# Drop the first column (does not have a meaning)
music_df = music_df.drop('Unnamed: 0', axis=1)

# Convert popularity column to binary values
music_df['popularity'] = music_df['popularity'].apply(lambda x: 1 if x >= music_df['popularity'].median() else 0)

# Select the features
X = music_df.drop('popularity', axis=1).values
y = music_df['popularity'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary with models to be evaluated
models = {"Logistic Regression": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier()}
results = []

for name, model in models.items():
    kf = KFold(n_splits=6, shuffle=True, random_state=12)

    # Cross validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf)

    # Performance evaluation
    print('Model: {} '.format(name))
    model.fit(X_train_scaled, y_train)
    print('Test set accuracy: {:.5f}'.format(model.score(X_test_scaled, y_test)))

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Compute the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print('Mean squared error: {:.5f} \n'.format(mse))

    # Append results to a list
    results.append(cv_scores)

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(results, labels=models.keys())
ax.set_ylabel('Accuracy')
plt.show()

# Note that the logistic regression model has the best performance but again,
# hyperparameter tuning is required to be done (future examples)

