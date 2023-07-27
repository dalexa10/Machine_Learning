""" Regularization
This is a technique to reduce overfitting by adding a penalty term to the loss function.

In a linear fitting, the loss function is the sum of the squared residuals.
The penalty term is the sum of the squared coefficients multiplied by a constant alpha.
The higher the alpha, the higher the penalty and the simpler the model.
Recall: pick alfa is analogous to picking k in k-NN.
Low alfa -> overfitting, high alfa -> underfitting.
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

# Load the data
sales_df = pd.read_csv('database/advertising_and_sales_clean.csv')
# print(sales_df.head())
print(sales_df.columns)

# Select the features
X = sales_df.drop('sales', axis=1).values
y = sales_df['sales'].values

# Plot the data
fig, ax = plt.subplots(2, 2)
ax[0, 0].scatter(X[:, 0], y, label='Data')
ax[0, 0].set_xlabel('TV')
ax[0, 0].set_ylabel('Sales')
ax[0, 0].legend()

ax[0, 1].scatter(X[:, 1], y, label='Data')
ax[0, 1].set_xlabel('Radio')
ax[0, 1].set_ylabel('Sales')
ax[0, 1].legend()

ax[1, 0].scatter(X[:, 2], y, label='Data')
ax[1, 0].set_xlabel('Social Media')
ax[1, 0].set_ylabel('Sales')
ax[1, 0].legend()

ax[1, 1].scatter(X[:, 3], y, label='Data')
ax[1, 1].set_xlabel('Influencer')
ax[1, 1].set_ylabel('Sales')
ax[1, 1].legend()
plt.tight_layout()
plt.show()

# Drop the influencer column, because it is not a numerical continuous feature
sales_df = sales_df.drop('influencer', axis=1)

# Select the features
X = sales_df.drop('sales', axis=1).values
y = sales_df['sales'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# ----------------------------------------------------------
#                   Ridge Regression
# ----------------------------------------------------------

scores_r = []
alpha_ls = [0.1, 1, 10, 100, 1000, 10000]

for alpha in alpha_ls:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    scores_r.append(ridge.score(X_test, y_test))

print(scores_r)

# ----------------------------------------------------------
#                   Lasso Regression
# ----------------------------------------------------------
scores_l = []
for alpha in alpha_ls:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    scores_l.append(lasso.score(X_test, y_test))

print(scores_l)

# Plot the scores
fig, ax = plt.subplots()
ax.plot(alpha_ls, scores_l, label='Lasso')
ax.plot(alpha_ls, scores_r, label='Ridge')
ax.set_xlabel('Alpha')
ax.set_ylabel('Score')
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.set_title('Performance of Lasso and Ridge Regression')
ax.legend(loc='best')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
#               Rank importance of features
# ----------------------------------------------------------
names = sales_df.drop('sales', axis=1).columns

lasso = Lasso(alpha=0.3)
lasso_coeff = lasso.fit(X, y).coef_

fig, ax = plt.subplots()
ax.bar(names, lasso_coeff)
ax.set_xlabel('Features')
ax.set_ylabel('Coefficients')
ax.set_yscale('log')
ax.set_title('Feature Importance')
plt.tight_layout()
plt.show()

print('It seems that TV plays a more important role in sales than the other features.')

