import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the regressor
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Compute score
print('Score: {}'.format(reg.score(X_test, y_test)))

# Compute the mean squared error
mse = mean_squared_error(y_test, y_pred, squared=False)
print('Mean squared error: {}'.format(mse))

# Plot features vs glucose
fig, ax = plt.subplots(2, 2)
ax[0, 0].scatter(X_test[:, 0], y_test, label='Data')
ax[0, 0].scatter(X_test[:, 0], y_pred, color='red', label='Prediction')
ax[0, 0].set_xlabel('Pregnancies')
ax[0, 0].set_ylabel('Blood Glucose')
ax[0, 0].legend()

ax[0, 1].scatter(X_test[:, 1], y_test, label='Data')
ax[0, 1].scatter(X_test[:, 1], y_pred, color='red', label='Prediction')
ax[0, 1].set_xlabel('Glucose')
ax[0, 1].set_ylabel('Blood Glucose')
ax[0, 1].legend()

ax[1, 0].scatter(X_test[:, 2], y_test, label='Data')
ax[1, 0].scatter(X_test[:, 2], y_pred, color='red', label='Prediction')
ax[1, 0].set_xlabel('Blood Pressure')
ax[1, 0].set_ylabel('Blood Glucose')
ax[1, 0].legend()

ax[1, 1].scatter(X_test[:, 3], y_test, label='Data')
ax[1, 1].scatter(X_test[:, 3], y_pred, color='red', label='Prediction')
ax[1, 1].set_xlabel('Skin Thickness')
ax[1, 1].set_ylabel('Blood Glucose')
ax[1, 1].legend()

plt.show()






