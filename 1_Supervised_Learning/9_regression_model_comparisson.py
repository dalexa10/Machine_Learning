"""
Guiding principles:
- Fewer features = simpler model and faster training
- Some models require large amounts of data to perform well

Interpretability:
- Linear models are very interpretable and easier to explain, the coefficients
can be explained

Flexibility:
- KNN has a lot of flexibility, it can learn non-linear relationships
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error


# Import data
music_df = pd.read_csv('database/music_clean.csv')

# Drop the first column (does not have a meaning)
music_df = music_df.drop('Unnamed: 0', axis=1)

# Dataset already clean
# music_df = music_df.dropna()

# Select the features
X = music_df.drop('energy', axis=1).values
y = music_df['energy'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary with models to be evaluated
models = {"Linear Regression": LinearRegression(),
          "Ridge": Ridge(alpha=1.),
          "Lasso": Lasso(alpha=0.1)}

results = []

for name, model in models.items():
    kf = KFold(n_splits=6, shuffle=True, random_state=42)

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

# Create a box plot for the results
fig, ax = plt.subplots()
ax.boxplot(results, labels=models.keys())
ax.set_ylabel('Cross Validation Score')
plt.show()


# Notes:
# The linear regression model is the best model for this dataset. However, a hyperparameter
# tuning could be performed to improve the performance of the Ridge and Lasso models.
# This latter will be done in future examples
# Note also that linear regression and ridge regression are fairly similar
