import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Import data from the csv file
df = pd.read_csv('database/telecom_churn_clean.csv')

# Select the features
X = df[['total_day_charge', 'total_eve_charge']].values

# Select the target
y = df['churn'].values

# Good practice to check the shape of the data
# print(X.shape, y.shape)

# Visualize the data
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y)
ax.set_xlabel('Total day charge')
ax.set_ylabel('Total eve charge')
plt.show()

# Split the data into training and test sets
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Create a k-NN classifier with variable neighbors
train_accuracy = {}
test_accuracy = {}
neighbors = np.arange(1, 50)

for neighbor in neighbors:
    # Instantiate the model
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    # Fit the model
    knn.fit(X_train, y_train)

    # Compute the accuracy
    train_accuracy[neighbor] = knn.score(X_train, y_train)
    test_accuracy[neighbor] = knn.score(X_test, y_test)

# Generate plot
fig, ax = plt.subplots()
ax.plot(neighbors, list(train_accuracy.values()), label='Training Accuracy')
ax.plot(neighbors, list(test_accuracy.values()), label='Testing Accuracy')
ax.set_xlabel('Number of Neighbors')
ax.set_ylabel('Accuracy')
ax.legend()
plt.show()


