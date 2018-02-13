#!/usr/bin/env python3
''' This is our first example of a machine learning algorithm.
    Here we import the digits dataset (A dataset of grayscale images)
    and attempt to classify each image from 0-9.  
'''


# Import the necessary modules from sklearn and matplotlib
from sklearn import svm
from sklearn import datasets
import matplotlib.pyplot as plt


# Load the dataset.
digits = datasets.load_digits()

#print (digits.data)
#print (digits.data.shape)

# We're going to use a support vector machine.  The gamma and C parameters are 
# known as hyperparameters.  Gamma is the learning rate. 
# 

clf = svm.SVC(gamma=0.0001, C=100)

# We're going to use 1757 out of the 1767 digits for our training data.  The remainder
# for our test data.
x,y = digits.data[:-10],digits.target[:-10]

# Fit a margin classification for the data. 
clf.fit(x,y)

# Test the classifier to see how well it predicts the number from the image. 
print ('Prediction: ', clf.predict(digits.data[[-4]]))

# Plot the image to see if what the classifier predicts matches what we expect.
plt.imshow(digits.images[-4], cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()
