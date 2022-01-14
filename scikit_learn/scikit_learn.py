# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Machine Learining to be implemented with a Scikit-Learn 

# **learning purpose**

# * Understanding various Algorithms of Machine Learning.
# * Skills in the use of Scikit-Learn libraries.
# * Understanding how to express data in a Scikit-Learn and how to divide data into datasets for traing and datasets for testing.

# [machine learning algorithms](https://blogs.sas.com/content/saskorea/2017/08/22/%EC%B5%9C%EC%A0%81%EC%9D%98-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EC%9D%84-%EA%B3%A0%EB%A5%B4%EA%B8%B0-%EC%9C%84%ED%95%9C-%EC%B9%98%ED%8A%B8/)

# **reinforcement learning developer community**

# * [reinfocement learning KR](https://github.com/reinforcement-learning-kr)
# * [aikorea/awesome-rl](https://github.com/aikorea/awesome-rl)

# **scikit-learn: choosing the right estimator**

# * [scikit-learn chart](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
# * [scikit-learn: main homepage](https://scikit-learn.org/stable/index.html)
# * [scikit-learn: api reference](https://scikit-learn.org/stable/modules/classes.html)

# !jupyter --paths

# version check
import sklearn
print(sklearn.__version__)
import pandas
print(pandas.__version__)

# **Data representation**

# * Scikit-learn offers datasets
#     * ndarray of NumPy
#     * DataFrame of Pandas
#     * Sparse Matrix of SciPy

# * In the Scikit-learn, data representation method
#     * Feature Matrix
#         * Input data.
#         * feature: It means individual observations represented by numerical, discrete, and boulian values in the data; in the Feature Matrix, the value corresponds to a columns.
#         * sample: Each input data; in the Feature Matrix, the value corresponds to a rows. 
#         * n_samples: Number of rows(number of samples).
#         * n_features: Number of columns(number of features).
#         * X: Generally, the variable name of the Featuce Matrix is notated as X.
#         * [n_samples, n_features] uses a two-dimensional array structure in the form of [row, column], which can be represented by ndarray of NumPy, DataFrame of Pandas, and Sparse Matrix of SciPy.
#     * Target Vector
#         * Label(correct answer) of input data.
#         * Target: called labels, target values; it refers to what you want to predict from the Feature Matrix.
#         * n_samples: The length of the vector(number of labels)
#         * No n_features in the Target Vector.
#         * y: Generally, the variable name of the Target Vector is notated as y.
#         * Target vectors are usually represented by one-dimensional vectors, which can be represented using the ndarray of NumPy, the Series of Pandas.
#         * (However, The Target Vector may not be represented in one dimension, in some cases).
#
#         
#         

# **!!! The n_sample of the Feature Matrix X and the n_sample of the Target Vector y must be same.**

# **1. Regression model**

import numpy as np
import matplotlib.pyplot as plt
r = np.random.RandomState(10)
x = 10 * r.rand(100)
y = 2 * x - 3 * r.rand(100)
plt.scatter(x,y)

# input data X
x.shape

# label data y
y.shape

# Shape of X, y is one-dimension vector.

# Model object generation for using Machine learning models in Sciki-learn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model

# +
# fit()
X = x.reshape(100,1)

model.fit(X,y)
# -

# predict()
x_new = np.linspace(-1, 11, 100)
X_new = x_new.reshape(100, 1)
y_new = model.predict(X_new)

X_ = x_new.reshape(-1, 1)
X_.shape

# **Performance Evaluation of Regression Model using RMSE**

# [Scikit-learn: Mean Squard Error](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error)

# +
from sklearn.metrics import mean_squared_error

error = np.sqrt(mean_squared_error(y, y_new))

print(error)
# -

plt.scatter(x,y, label='input data')
plt.plot(X_new, y_new, color='red', label='regression line')

# : Dots and a regression line almost match in the graph

# **2. datasets.load_wine()**

from sklearn.datasets import load_wine
data = load_wine()
type(data)

# bunch data type is similar to Python's Dictionary 
print(data)

# bunch type also can use key of Python's Dictionary
data.keys()

data.data.shape

# 13 feature and 178 data of Feature Matrix

# dimension check    ndim
data.data.ndim

# target vector: key values
data.target

data.target.shape

# feature names check
data.feature_names

# number of feature
len(data.feature_names)

# : The number of feature_names and the number of n_features(column) of the Feature Matrix match

# target_names
data.target_names

# DESCR: explanation about datasets
print(data.DESCR)

# **3. DataFrame**

# +
import pandas as pd

pd.DataFrame(data.data, columns=data.feature_names)
# -
# **4. Machine learning**


X = data.data
y = data.target

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# model train
model.fit(X,y)

# prediction
y_pred = model.predict(X)

# +
# performance estimation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# 타겟 벡터 즉 라벨인 변수명 y, 예측값 y_pred을 각각 인자로 넣는다.
print(classification_report(y, y_pred))
# output accuracy
print("accuracy = ", accuracy_score(y, y_pred))
# -

# : The same data training and predictions have shown a 100% accuracy

# **5. Separate train data and test data**

# ![%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-14%20150053.jpg](attachment:%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-14%20150053.jpg)

from sklearn.datasets import load_wine
data = load_wine()
print(data.data.shape)
print(data.target.shape)

# : The total number of data is 178; 8 to 2 divides the feature matrix and the target vector. 80% of 178 are 142.4, but 142 are expressed as integers, and the training data are divided into 36.

# ![%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-14%20152216.jpg](attachment:%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-14%20152216.jpg)

# The feature matrix and target vector are ndarray type,
# so use the slicing of the numpy

# separating data set
X_train = data.data[:142]
X_test = data.data[142:]
y_train = data.target[:142]
y_test = data.target[142:]
print(X_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# +
# training
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# +
# accuracy estimation
from sklearn.metrics import accuracy_score

print("accuracy: ", accuracy_score(y_test, y_pred))
# -

# **6. Use the train_test_split() and separate**

# +
from sklearn.model_selection import train_test_split

result = train_test_split(X, 
                          y, 
                          test_size=0.2, 
                          random_state=42)

print(type(result))
print(len(result))
# -

result[0].shape

result[1].shape

result[2].shape

result[3].shape

# unpacking
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# +
# total code

data = load_wine()

X_train, X_test, y_train, y_test = train_test_split(data.data, 
                                                    data.target, 
                                                    test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("정답률=", accuracy_score(y_test, y_pred))
