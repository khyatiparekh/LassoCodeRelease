
import numpy as np
import Lasso
from sklearn import preprocessing

# Creating a dummy dataset
x1 = np.append( np.repeat( 70, 30), np.repeat( 100, 30 ) )
x2 = np.append( np.repeat( 15, 30), np.repeat( 20, 30 ) )
X = np.column_stack((x1, x2))
y = np.append( np.repeat( 1, 30), np.repeat( -1,30 ) )

beta_init = np.zeros(np.size(X, 1))
max_iter = 500
betas_cyclic = Lasso.cycliccoorddescent(beta_init, X, y, 1, max_iter)
print('Optimal beta from cyclic coordinate descent:\n', betas_cyclic[-1, :])