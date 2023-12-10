import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

regressor = DecisionTreeRegressor(max_depth=5)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])
print(y_pred)

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Decision Tree Regressor')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
