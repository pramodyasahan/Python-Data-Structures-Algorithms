import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Loading the dataset
dataset = pd.read_csv('student-mat.csv', delimiter=';')

# Selecting specific columns as features based on their indices
features_index = [8, 9, 13, 29, 30, 31]
X = dataset.iloc[:, features_index].values

# Selecting the last column as the target variable (final grade)
y = dataset.iloc[:, -1].values

# Applying OneHotEncoder to the first two categorical columns
# and keeping the rest of the columns unchanged
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating and training the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Making predictions on the test set
y_pred = regressor.predict(X_test)

# Setting print options for precision and printing the predicted and actual values
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Creating an index array (x-axis)
index = range(len(y_pred))

# Plotting y_pred and y_test
plt.plot(index, y_pred, color='red', label='Predicted Grades')
plt.plot(index, y_test, color='blue', label='Actual Grades')

# Adding title and labels
plt.title('Comparison of Predicted and Actual Grades')
plt.xlabel('Index of Samples')
plt.ylabel('Grades')
plt.legend()

# Show the plot
plt.show()
