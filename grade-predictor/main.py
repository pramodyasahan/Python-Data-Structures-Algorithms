import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('student-mat.csv', delimiter=';')
features_index = [8, 9, 13, 29, 30, 31]
X = dataset.iloc[:, features_index].values
y = dataset.iloc[:, -1].values


