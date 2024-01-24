import os
import pickle
import numpy as np
import random
class ProcessAll:
    def __init__(self, classes, resultPath, pcapNum):
        self.pcapNum = pcapNum
        y_train = []
        y_test = []
        y_valid = []
        for i in range(len(classes)):
            tmptrain = classes[i] * 1800
            y_train += tmptrain
            tmptest = classes[i] * 100
            tmpvalid = classes[i] * 100
            y_test += tmptest
            y_valid += tmpvalid
        self.y_train = y_train
        self.y_test = y_test
        self.y_valid = y_valid
        self.x_train = []
        self.x_test = []
        self.x_valid = []
        self.resultPath = resultPath
        self.trainNameX = "trainDataX.pkl"
        self.testNameX = "testDataX.pkl"
        self.validNameX = "validDataX.pkl"
        self.trainNameY = "trainDataY.pkl"
        self.testNameY = "testDataY.pkl"
        self.validNameY = "validDataY.pkl"



    # resultPath = "pcapDirections/"

    def create_pkl(self):
        for j in range(len(self.pcapNum)):
            fileList = []

            for dirname, dirnames, filenames in os.walk(self.resultPath+str(self.pcapNum[j])):
                if len(filenames) != 0:
                    for filename in filenames:
                        fileList.append(filename)

            fileCount = 0

            for name in fileList:
                valueArray = []

                if fileCount > 1999:
                    break
                # test set count 100
                if 1799 < fileCount < 1900:
                    with open(self.resultPath + name, 'r') as f:
                        line = f.readline()
                        valueArray.append(int(line))
                        while line:
                            line = f.readline()
                            if not line:
                                break
                            valueArray.append(int(line))
                    self.x_test += valueArray
                    fileCount += 1
                    continue
                # valid set count 100
                if 1899 < fileCount < 2000:
                    with open(self.resultPath + name, 'r') as f:
                        line = f.readline()
                        valueArray.append(int(line))
                        while line:
                            line = f.readline()
                            if not line:
                                break
                            valueArray.append(int(line))
                    self.x_valid += valueArray
                    fileCount += 1
                    continue
                # train set count 1800
                with open(self.resultPath + name, 'r') as f:
                    line = f.readline()
                    valueArray.append(int(line))
                    while line:
                        line = f.readline()
                        if not line:
                            break
                        valueArray.append(int(line))
                self.x_train += valueArray
                fileCount += 1

        # checking the length of the data and making it fixed 5000 length
        for i in range(len(self.x_train)):
            length = len(self.x_train[i])
            if length <= 5000:
                lengthDiff = 5000 - length
                for j in range(lengthDiff):
                    self.x_train[i].append(0)
            else:
                self.x_train[i] = self.x_train[i][0:5000]

        for i in range(len(self.x_test)):
            length = len(self.x_test[i])
            if length <= 5000:
                lengthDiff = 5000 - length
                for j in range(lengthDiff):
                    self.x_test[i].append(0)
            else:
                self.x_test[i] = self.x_test[i][0:5000]

        for i in range(len(self.x_valid)):
            length = len(self.x_valid[i])
            if length <= 5000:
                lengthDiff = 5000 - length
                for j in range(lengthDiff):
                    self.x_valid[i].append(0)
            else:
                self.x_train[i] = self.x_train[i][0:5000]

        # let's shuffle training set [will take too much memory. let's shuffle during training]
        # tmp = list(zip(self.x_train, self.y_train))
        # random.shuffle(tmp)
        # self.x_train, self.y_train = list(zip(*tmp))

        #Dump the dataset
        trainOutX = open(self.trainNameX, 'wb')
        print("Dumping Training Data X")
        pickle.dump(self.x_train, trainOutX)

        trainOutY = open(self.trainNameY, 'wb')
        print("Dumping Training Data Y")
        pickle.dump(self.y_train, trainOutY)

        testOutX = open(self.testNameX, 'wb')
        print("Dumping Test Data X")
        pickle.dump(self.x_test, testOutX)

        testOutY = open(self.testNameY, 'wb')
        print("Dumping Test Data Y")
        pickle.dump(self.y_test, testOutY)

        validOut = open(self.validNameX, 'wb')
        print("Dumping Validation Data X")
        pickle.dump(self.x_valid, validOut)

        validOutY = open(self.validNameY, 'wb')
        print("Dumping Validation Data X")
        pickle.dump(self.y_valid, validOutY)

        print("Dataset Creation Succeeded!")