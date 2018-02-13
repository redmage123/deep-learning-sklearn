#!/usr/bin/env python3
from sklearn.neural_network import  MLPClassifier
import numpy as np

xs =  np.array([
                 0,0,
                 0,1,
                 1,0,
                 1,1
               ]).reshape(4,2)

ys = np.array([0,1,1,0]).reshape(4,)

model = MLPClassifier(activation = 'relu')
model.fit (xs,ys)

print ('score: ',model.score(xs,ys))


