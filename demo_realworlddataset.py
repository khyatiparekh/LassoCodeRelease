import pandas as pd
import matplotlib.pyplot as mpl
import Lasso
import numpy as np
from sklearn import preprocessing, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

hitters = pd.read_csv('Hitters.csv', sep=',', header=0)


hitters = hitters.dropna()
hitters.head(5)

X = hitters.drop('Salary', axis=1)  # Axis denotes either the rows (0) or the columns (1)
y = hitters.Salary


X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = np.array(y - np.mean(y))


beta_init = np.zeros(np.size(X, 1))
max_iter = 500
betas_cyclic = Lasso.cycliccoorddescent(beta_init, X, y, 1, max_iter)
print('Optimal beta from cyclic coordinate descent:\n', betas_cyclic[-1, :])