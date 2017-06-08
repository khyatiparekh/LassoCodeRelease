
import pandas as pd
import Lasso
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

hitters = pd.read_csv('Hitters.csv')
hitters = hitters.dropna()
hitters.head(5)

X = hitters.drop('Salary', axis=1)  # Axis denotes either the rows (0) or the columns (1)
y = hitters.Salary


X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = np.array(y - np.mean(y))

print("Running Lasso Regression with Coordinate Descent..")
# Running Fast Gradient Descent
beta_init = np.zeros(np.size(X, 1))
max_iter = 1000
betas_cyclic = Lasso.cycliccoorddescent(beta_init, X, y, 1, max_iter)
print('Optimal beta from cyclic coordinate descent:\n', betas_cyclic[-1, :])

print("\n\n\nRunning sklearn's version of lasso regression")
lasso = LassoCV(fit_intercept=False).fit(X, y)
lambda_opt = lasso.alpha_*2  # Note scikit-learn's objective function is different from ours
beta_star = lasso.coef_

print("Coefficient Values using sklearn = ", beta_star)













