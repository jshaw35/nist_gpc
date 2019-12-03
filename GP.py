# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:33:24 2019

@author: jks7

Taking GP example from scikit-learn to try and understand
"""

from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

import numpy as np

def main():
    X, y = make_friedman2(n_samples=500, noise=0, random_state=0) # This is the test data we are fitting
    print(type(X))
    print(np.shape(X))  # 500 samples each with 4 variables
    print(type(y))
    print(np.shape(y))  # 500 outputs
    
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer= 'fmin_l_bfgs_b',
            random_state=0).fit(X, y)
    gpr_score = gpr.score(X, y)         # R2 of the prediction
    print('Prediction of R^2: %f' % gpr_score)
    
    print("Shape of thing ", np.shape(X[:2,:]))
    print("Thing ", X[:2,:])
    gpr_predict = gpr.predict(X[:2,:], return_std=True)
    print(gpr_predict)
    # Now how to I get it to predict another spot to sample?

if __name__ == '__main__':
    main()