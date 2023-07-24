import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

# Making predictions from single feature
X_bmi = X[:, 4].reshape(-1, 1)

# Create the regressor
reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)

# Plot the bmi vs diabetes
fig, ax = plt.subplots()
ax.scatter(X_bmi, y, label='Data')
ax.plot(X_bmi, predictions, color='red', label='Prediction')
ax.set_xlabel('Body Mass Index')
ax.set_ylabel('Blood Glucose')
ax.legend()
plt.show()



