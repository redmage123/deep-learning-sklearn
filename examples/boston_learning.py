#!/usr/bin/env python3
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
import numpy as np
import pylab as pl
from sklearn.datasets import load_boston
boston = load_boston()
print (boston.feature_names)
print (boston.data.shape)
print (boston.target.shape)
np.set_printoptions(precision=2, linewidth=120, suppress=True, edgeitems=4)
print (boston.data)
x = np.array([np.concatenate((v,[1])) for v in boston.data])
y = boston.target
print (y[:10])
linreg = LinearRegression()
p = linreg.predict(x)
err = abs(p-y)
print (err[:10])
rmse_train = np.sqrt(total_error/len(p))
print (rmse_train)
print ('Regression Coefficients: \n',linreg.coef)

