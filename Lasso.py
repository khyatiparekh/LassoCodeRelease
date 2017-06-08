import numpy as np
import pandas as pd

def soft_threshold(c, lambduh):
    if c < -lambduh:
        return c+lambduh
    elif c > lambduh:
        return c-lambduh
    else:
        return 0





def coord_descent_solution(j, X, lambduh, beta, a, xy):

    c_j = 2*(xy[j] - np.sum(X[:,j]*(np.dot(X, beta) - X[:, j]*beta[j])))

    return soft_threshold(c_j, lambduh)/a[j]




def computeobj(beta, X, y, lambduh):
    return 1/np.size(X, 0)*sum((y-np.dot(X, beta))**2) + lambduh*sum(abs(beta))




def cycliccoorddescent(beta, X, y, lambduh, max_iter):
    d = np.size(X, 1)
    a = 2*np.sum(X**2, axis=0)
    xy = np.dot(X.T, y)
    all_betas = beta
    for i in range(max_iter):
        for j in range(d):
            beta[j] = coord_descent_solution(j, X, lambduh, beta, a, xy)
        all_betas = np.vstack((all_betas, beta))

    return all_betas

