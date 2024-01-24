"""A script to check the prediction of our trained classifier"""
import numpy as np
import pickle
import torch
from websiteChecker import WebsiteIndex as wc
import WebsiteInstance as wi


# load batch data
def load_replaced_test(webName):
    dataset_dir = 'dataSet/NoDef/'
    with open(dataset_dir + 'testReplace' + webName, 'rb', ) as dt:
        X_test_replace = np.array(pickle.load(dt, encoding='latin1'))

    X_test = X_test_replace[:, np.newaxis, :]

    X_test = torch.Tensor(X_test)

    return X_test


def load__test():
    dataset_dir = 'dataSet/NoDef/'

    with open(dataset_dir + 'validDataX.pkl', 'rb') as dt:
        X_test = pickle.load(dt, encoding='bytes')  # data is numpy arrays
    with open(dataset_dir + 'validDataY.pkl', 'rb') as dt:
        y_test = pickle.load(dt, encoding='bytes')

    X_test = X_test[:, np.newaxis, :]

    X_test = torch.Tensor(X_test)

    return X_test


def load_test2():
    dataset_dir = 'dataSet/NoDef/'

    with open(dataset_dir + 'testDataX20.pkl', 'rb') as dt:
        X_test = pickle.load(dt, encoding='bytes')  # data is numpy arrays
    with open(dataset_dir + 'testDataY20.pkl', 'rb') as dt:
        y_test = pickle.load(dt, encoding='bytes')

    X_test = X_test[:, np.newaxis, :]

    X_test = torch.Tensor(X_test)

    return X_test


d1 = load_replaced_test('Google')
print(d1.shape)
data = []
data.append(d1)

webCheck = wc(1, data, 20, [19])
print(webCheck.classify())

# d1 = load_replaced_test('amazon')
# data = []
# data.append(d1)

# webCheck = wc(1, data, 20, [5])
# print(webCheck.classify())

# d1 = load_replaced_test('youtube')
# data = []
# data.append(d1)

# webCheck = wc(1, data, 20, [1])
# print(webCheck.classify())

# d1 = load_replaced_test('wikpedia')
# data = []
# data.append(d1)

# webCheck = wc(1, data, 20, [7])
# print(webCheck.classify())

# d1 = load_replaced_test('tmall')
# data = []
# data.append(d1)

# webCheck = wc(1, data, 20, [16])
# print(webCheck.classify())

# d1 = load_replaced_test('facebook')
# data = []
# data.append(d1)

# webCheck = wc(1, data, 20, [3])
# print(webCheck.classify())

# with open('dataSet/NoDef/' + 'X_test_NoDef.pkl', 'rb') as dt:
# X_test = np.array(pickle.load(dt, encoding='bytes'))
# print(X_test[100])
# with open('dataSet/NoDef/' + 'y_test_NoDef.pkl', 'rb') as dt:
# y_test = np.array(pickle.load(dt, encoding='bytes'))
# X_test = X_test[200:208]
# X_test = X_test[:, np.newaxis, :]

# X_test = torch.Tensor(X_test)

# y_test = torch.Tensor(y_test).long()
# data = []
# data.append(X_test)
# wc2 = wc(1, data)
# print(wc2.classify())
# print(y_test[200:208])
# testReplaceGoogle
# with open('dataSet/NoDef/' + 'testReplaceGoogle', 'rb') as dt:
# X_test = np.array(pickle.load(dt, encoding='bytes'))
# print(X_test[100])
# with open('dataSet/NoDef/' + 'y_test_NoDef.pkl', 'rb') as dt:
# y_test = np.array(pickle.load(dt, encoding='bytes'))
# X_test = X_test[1:50]
# X_test = X_test[:, np.newaxis, :]

# X_test = torch.Tensor(X_test)
# data = []
# data.append(X_test)
# wc2 = wc(1, data)
# print(wc2.classify())

d1 = load_test2()
data = []
for i in range(10):
    data.append(d1[i * 20:i * 20 + 20])
data.append(d1)

webCheck = wc(10, data, 20, [0, 1, 3, 5, 7, 14, 15, 17, 18, 19])
print(webCheck.classify())

with open('dataSet/NoDef/' + 'testReplacestackoverflow', 'rb') as dt:
    X_test = np.array(pickle.load(dt, encoding='bytes'))
X_test = X_test[1:50]
X_test = X_test[:, np.newaxis, :]

X_test = torch.Tensor(X_test)
data = []
data.append(X_test)
wc2 = wc(1, data, 20, [17])
print(wc2.classify())

# check single website instance given the feature file with direction
fileName = "prac4ye.txt"  # direction feature for youtube sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc3 = wc(1, data, 20, [1])
print(wc3.classify())

fileName = "testing/twittertestFeature.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [19])
print(wc4.classify())

fileName = "testing/twitter/46/directions/46/8.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [19])
print(wc4.classify())

fileName = "testing/twitter/46/directions/46/7.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [19])
print(wc4.classify())
fileName = "testing/twitter/46/directions/46/27.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [19])
print(wc4.classify())

fileName = "testing/facebook/directions/4/0.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [3])
print(wc4.classify())

fileName = "testing/reddit/directions/23/0.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [14])
print(wc4.classify())

fileName = "testing/stackOverflow/directions/38/8.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [17])
print(wc4.classify())
fileName = "testing/stackOverflow/directions/38/9.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [17])
print(wc4.classify())
fileName = "testing/stackOverflow/directions/38/10.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [17])
print(wc4.classify())
fileName = "testing/stackOverflow/directions/38/11.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [17])
print(wc4.classify())

fileName = "testing/stackOverflow/directions/38/12.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [17])
print(wc4.classify())

fileName = "testing/stackOverflow/directions/38/13.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [17])
print(wc4.classify())

fileName = "testing/stackOverflow/directions/38/14.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [17])
print(wc4.classify())

fileName = "testing/stackOverflow/directions/38/15.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [17])
print(wc4.classify())

fileName = "testing/stackOverflow/directions/38/16.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [17])
print(wc4.classify())

fileName = "testing/stackOverflow/directions/38/17.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [17])
print(wc4.classify())

fileName = "testing/stackOverflow/directions/38/18.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [17])
print(wc4.classify())

fileName = "testing/stackOverflow/directions/38/19.txt"  # direction feature for twitter sample
x = wi.generate_data(fileName)  # generate data format for the classifier
data = [x]
wc4 = wc(1, data, 20, [17])
print(wc4.classify())
