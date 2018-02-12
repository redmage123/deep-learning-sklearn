#!/usr/bin/env python3

from sklearn.linear_model import Perceptron
import numpy as np

T,F = 1.,0.
bias = 1

train_x = [
              [T,T],
              [T,F],
              [F,T],
              [F,F]
          ]

train_or_y = [
              [T],
              [T],
              [T],
              [F]
          ]

train_and_y = [
              [T],
              [F],
              [F],
              [F]
          ]

W = np.random.normal([3,1])

train_or_y = np.ravel(train_or_y)
train_and_y = np.ravel(train_and_y)
my_or  = Perceptron(class_weight='balanced')
my_and = Perceptron(class_weight='balanced')
my_or.coef_ = W
my_or.intercept_ = [bias]
my_and.coef_ = W
my_and.intercept_ = [bias]
my_or.fit(train_x,train_or_y)
my_and.fit(train_x,train_and_y)

p = my_or.predict([[T,T]])
print (p)
p = my_or.predict([[F,F]])
print (p)

p = my_and.predict([[T,T]])
print (p)
p = my_and.predict([[F,F]])
print (p)
