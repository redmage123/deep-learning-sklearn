#!/usr/bin/env python3

from sklearn import model_selection, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys

# Load the Iris dataset
iris = datasets.load_iris()

iris_df = pd.DataFrame(iris.data,columns=['sepal-length','sepal-width','petal-length','petal-width'])


# Let's do some pre-processing here. 
# First let's examine our data.


print (iris_df.head(10))
print (iris_df.describe())
iris_df.plot(kind='box',subplots=True, layout = (2,2), sharex=False, sharey=False)
plt.show()

iris_df['Class'] = iris.target
print (iris_df.groupby('Class').size())
iris_df.hist()
plt.show()


# Split the data set into the training data (0.7) and testing
# (0.3).  Also, shuffle the data

X_train,X_test, y_train,y_test = model_selection.train_test_split(iris.data,iris.target,test_size = 0.3,shuffle=True)

models = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))


#lr = LogisticRegression()
for name,model in models:
    current_model = model

# Fit the training data to the model.
    current_model.fit(X_train,y_train)

# Test out the first ten elements of the testing set to see
# How well our model predicts. 

for iris_test_x, iris_test_y  in zip(X_test,y_test):
    for name,model in models:
        current_model = model

        p = current_model.predict([iris_test_x])
        print ('Predicted class: %s, Actual class: %s' % (p,iris_test_y))
        print ('\n')

for name,model in models:
        expected = iris.target
        predicted = current_model.predict(iris.data)
        print ("Model = ", name,"\n",metrics.classification_report(expected,predicted))
        print ("Model = ", name, "\n", metrics.confusion_matrix(expected,predicted))
