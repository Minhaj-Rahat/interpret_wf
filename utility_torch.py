import torch
import copy
import numpy as np
import pickle
import torch.nn as nn


def loadDataTorch():
    print("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = 'dataSet/NoDef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as dt:
        X_train = np.array(pickle.load(dt, encoding='bytes'))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as dt:
        y_train = np.array(pickle.load(dt, encoding='bytes'))

    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as dt:
        X_valid = np.array(pickle.load(dt, encoding='bytes'))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as dt:
        y_valid = np.array(pickle.load(dt, encoding='bytes'))

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as dt:
        X_test = np.array(pickle.load(dt, encoding='bytes'))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as dt:
        y_test = np.array(pickle.load(dt, encoding='bytes'))

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    X_train = X_train[:, np.newaxis, :]
    X_valid = X_valid[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    X_train = torch.Tensor(X_train)
    X_valid = torch.Tensor(X_valid)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train).long()
    y_valid = torch.Tensor(y_valid).long()
    y_test = torch.Tensor(y_test).long()

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def loadDataMine():
    print("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = 'dataSet/NoDef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'trainDataX.pkl', 'rb') as dt:
        X_train = pickle.load(dt, encoding='bytes')
    with open(dataset_dir + 'trainDataY.pkl', 'rb') as dt:
        y_train = pickle.load(dt, encoding='bytes')

    # Load validation data
    with open(dataset_dir + 'validDataX.pkl', 'rb') as dt:
        X_valid = pickle.load(dt, encoding='bytes')
    with open(dataset_dir + 'validDataY.pkl', 'rb') as dt:
        y_valid = pickle.load(dt, encoding='bytes')

    # Load testing data
    with open(dataset_dir + 'testDataX.pkl', 'rb') as dt:
        X_test = pickle.load(dt, encoding='bytes')
    with open(dataset_dir + 'testDataY.pkl', 'rb') as dt:
        y_test = pickle.load(dt, encoding='bytes')

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    X_train = X_train[:, np.newaxis, :]
    X_valid = X_valid[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    X_train = torch.Tensor(X_train)
    X_valid = torch.Tensor(X_valid)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train).long()
    y_valid = torch.Tensor(y_valid).long()
    y_test = torch.Tensor(y_test).long()

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def loadTestDataTorch():
    print("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = 'dataSet/NoDef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as dt:
        X_test = np.array(pickle.load(dt, encoding='bytes'))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as dt:
        y_test = np.array(pickle.load(dt, encoding='bytes'))

    print("Data dimensions:")
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    X_test = X_test[:, np.newaxis, :]

    X_test = torch.Tensor(X_test)

    y_test = torch.Tensor(y_test).long()

    return X_test, y_test


def loadReplacedData():
    dataset_dir = 'dataSet/NoDef/'
    with open(dataset_dir + 'trainReplaceGoogle', 'rb', ) as dt:
        X_train_replace = np.array(pickle.load(dt, encoding='latin1'))

    with open(dataset_dir + 'testReplaceGoogle', 'rb', ) as dt:
        X_test_replace = np.array(pickle.load(dt, encoding='latin1'))

    with open(dataset_dir + 'validReplaceGoogle', 'rb', ) as dt:
        X_valid_replace = np.array(pickle.load(dt, encoding='latin1'))

    print("Loading non-defended dataset for closed-world scenario")

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as dt:
        X_train = np.array(pickle.load(dt, encoding='bytes'))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as dt:
        y_train = np.array(pickle.load(dt, encoding='bytes'))

    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as dt:
        X_valid = np.array(pickle.load(dt, encoding='bytes'))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as dt:
        y_valid = np.array(pickle.load(dt, encoding='bytes'))

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as dt:
        X_test = np.array(pickle.load(dt, encoding='bytes'))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as dt:
        y_test = np.array(pickle.load(dt, encoding='bytes'))

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    trainCount = 0

    testCount = 0

    validCount = 0

    for i in range(len(y_train)):
        if y_train[i] == 0:
            # len(X_train_replace[trainCount])
            X_train[i] = X_train_replace[trainCount]
            # print(trainCount)
            trainCount += 1

    for i in range(len(y_test)):
        if y_test[i] == 0:
            X_test[i] = X_test_replace[testCount]
            testCount += 1

    for i in range(len(y_valid)):
        if y_valid[i] == 0:
            X_valid[i] = X_valid_replace[validCount]
            validCount += 1

    X_train = X_train[:, np.newaxis, :]
    X_valid = X_valid[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    X_train = torch.Tensor(X_train)
    X_valid = torch.Tensor(X_valid)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train).long()
    y_valid = torch.Tensor(y_valid).long()
    y_test = torch.Tensor(y_test).long()

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def loadReplacedDataV2(webName, classNum, inList):
    dataset_dir = 'dataSet/NoDef/'
    X_train_replace = []
    X_test_replace = []
    X_valid_replace = []

    for i in range(len(classNum)):
        with open(dataset_dir + 'trainReplace' + webName[i], 'rb', ) as dt:
            X_train_replace.append(np.array(pickle.load(dt, encoding='latin1')))

        with open(dataset_dir + 'testReplace' + webName[i], 'rb', ) as dt:
            X_test_replace.append(np.array(pickle.load(dt, encoding='latin1')))

        with open(dataset_dir + 'validReplace' + webName[i], 'rb', ) as dt:
            X_valid_replace.append(np.array(pickle.load(dt, encoding='latin1')))

        # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as dt:
        X_train = np.array(pickle.load(dt, encoding='bytes'))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as dt:
        y_train = np.array(pickle.load(dt, encoding='bytes'))

    # inList = [True, False]
    #for i in range(len(classNum)):
        #if classNum in y_train:
            #inList.append(True)
        #else:
            #inList.append(False)
    #print(inList)
    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as dt:
        X_valid = np.array(pickle.load(dt, encoding='bytes'))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as dt:
        y_valid = np.array(pickle.load(dt, encoding='bytes'))

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as dt:
        X_test = np.array(pickle.load(dt, encoding='bytes'))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as dt:
        y_test = np.array(pickle.load(dt, encoding='bytes'))

    print("Loading non-defended dataset for closed-world scenario")
    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    for j in range(len(inList)):

        if inList[j] == True:
            trainCount = 0

            testCount = 0

            validCount = 0
            for i in range(len(y_train)):
                if y_train[i] == classNum[j]:
                    # len(X_train_replace[trainCount])
                    X_train[i] = X_train_replace[j][trainCount]
                    # print(trainCount)
                    trainCount += 1

            for i in range(len(y_test)):
                if y_test[i] == classNum[j]:
                    X_test[i] = X_test_replace[j][testCount]
                    testCount += 1

            for i in range(len(y_valid)):
                if y_valid[i] == classNum[j]:
                    X_valid[i] = X_valid_replace[j][validCount]
                    validCount += 1

        else:
            # print("Appending zeros to X data")
            # X_train
            #n = np.zeros(X_train_replace[j].shape)
            # np.append(X_train, n, axis=0)
            X_train = np.append(X_train, X_train_replace[j], axis=0)
            X_test = np.append(X_test, X_test_replace[j], axis=0)
            X_valid = np.append(X_valid, X_valid_replace[j], axis=0)

            for i in range(X_train_replace[j].shape[0]):
                #np.append(X_train, X_train_replace[j][i], axis=0)
                y_train = np.append(y_train, classNum[j])
            for i in range(X_test_replace[j].shape[0]):
                #np.append(X_test, X_test_replace[j][i], axis=0)
                y_test = np.append(y_test, classNum[j])
            for i in range(X_valid_replace[j].shape[0]):
                #np.append(X_valid, X_valid_replace[j][i], axis=0)
                y_valid = np.append(y_valid, classNum[j])

    print("New Data Dimension")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    X_train = X_train[:, np.newaxis, :]
    X_valid = X_valid[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    X_train = torch.Tensor(X_train)
    X_valid = torch.Tensor(X_valid)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train).long()
    y_valid = torch.Tensor(y_valid).long()
    y_test = torch.Tensor(y_test).long()

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def load_replaced_test():
    dataset_dir = 'dataSet/NoDef/'
    with open(dataset_dir + 'testReplaceGoogle', 'rb', ) as dt:
        X_test_replace = np.array(pickle.load(dt, encoding='latin1'))

    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as dt:
        X_test = np.array(pickle.load(dt, encoding='bytes'))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as dt:
        y_test = np.array(pickle.load(dt, encoding='bytes'))

    testCount = 0

    for i in range(len(y_test)):
        if y_test[i] == 0:
            X_test[i] = X_test_replace[testCount]
            testCount += 1

    X_test = X_test[:, np.newaxis, :]

    X_test = torch.Tensor(X_test)

    y_test = torch.Tensor(y_test).long()

    return X_test, y_test


def toconv(layer, i):
    newlayer = None

    if i == 25:
        m, n = 256, layer.weight.shape[0]
        newlayer = torch.nn.Conv1d(m, n, 20)
        newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n, m, 20))
        # newlayer = torch.nn.Conv1d(m, n, 13)
        # newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n, m, 13))

    else:
        m, n = layer.weight.shape[1], layer.weight.shape[0]
        newlayer = torch.nn.Conv1d(m, n, 1)
        newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n, m, 1))

    newlayer.bias = torch.nn.Parameter(layer.bias)

    return newlayer


def fuse_conv(conv, bn):
    b = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).reshape(conv.out_channels, 1, 1)
    weight_new = conv.weight * b
    bias_new = bn.bias + bn.weight * (conv.bias - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps)

    fused_conv = torch.nn.Conv1d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        padding_mode=conv.padding_mode
    )
    fused_conv.weight = torch.nn.Parameter(weight_new)
    fused_conv.bias = torch.nn.Parameter(bias_new)
    return fused_conv

    # running_mean, running var


def fuse_linear(lin, bn):
    b = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).reshape(lin.weight.shape[0], 1)
    weight_new = lin.weight * b
    bias_new = bn.bias + bn.weight * (lin.bias - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps)

    fused_lin = torch.nn.Linear(lin.in_features, lin.out_features)
    fused_lin.weight = torch.nn.Parameter(weight_new)
    fused_lin.bias = torch.nn.Parameter(bias_new)
    return fused_lin


def layer_new(layer, g):
    layer = copy.deepcopy(layer)
    try:
        layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError:
        pass

    try:
        layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError:
        pass

    return layer


# Custom initilizer for torch_model
def torch_initializer(m):
    #if isinstance(m, nn.Conv1d):
        #nn.init.xavier_uniform_(m.weight.data)
        # nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight.data)
