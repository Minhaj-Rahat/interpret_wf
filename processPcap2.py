import os
import pickle


class processPcap:
  def __init__(self, pcapNumber, webName):
    self.pcapNumber = pcapNumber
    self.webName = webName
    self.resultPath = "directionFiltered/"+str(self.pcapNumber)+"/"
    self.trainName = 'trainReplace' + self.webName
    self.testName = 'testReplace' + self.webName
    self.validName = 'validReplace' + self.webName


  def create_pkl(self):
    fileList = []

    for dirname, dirnames, filenames in os.walk(self.resultPath):
      if len(filenames) != 0:
        for filename in filenames:
          fileList.append(filename)

    fileCount = 0

    trainSet = []
    validSet = []
    testSet = []

    for name in fileList:
      valueArray = []

      if fileCount > 999:
        break

      if 799 < fileCount < 900:
        with open(self.resultPath + name, 'r') as f:
          line = f.readline()
          valueArray.append(int(line))
          while line:
            line = f.readline()
            if not line:
              break
            valueArray.append(int(line))
        testSet.append(valueArray)
        fileCount += 1
        continue

      if 899 < fileCount < 1000:
        with open(self.resultPath + name, 'r') as f:
          line = f.readline()
          valueArray.append(int(line))
          while line:
            line = f.readline()
            if not line:
              break
            valueArray.append(int(line))
        validSet.append(valueArray)
        fileCount += 1
        continue

      with open(self.resultPath + name, 'r') as f:
        line = f.readline()
        valueArray.append(int(line))
        while line:
          line = f.readline()
          if not line:
            break
          valueArray.append(int(line))
      trainSet.append(valueArray)
      fileCount += 1

    for i in range(len(trainSet)):
      length = len(trainSet[i])
      if length <= 5000:
        lengthDiff = 5000 - length
        for j in range(lengthDiff):
          trainSet[i].append(0)
      else:
        trainSet[i] = trainSet[i][0:5000]

    for i in range(len(testSet)):
      length = len(testSet[i])
      if length <= 5000:
        lengthDiff = 5000 - length
        for j in range(lengthDiff):
          testSet[i].append(0)
      else:
        testSet[i] = testSet[i][0:5000]

    for i in range(len(validSet)):
      length = len(validSet[i])
      if length <= 5000:
        lengthDiff = 5000 - length
        for j in range(lengthDiff):
          validSet[i].append(0)
      else:
        validSet[i] = validSet[i][0:5000]

    trainOut = open(self.trainName, 'wb')
    print("Dumping Training Data")
    pickle.dump(trainSet, trainOut)

    testOut = open(self.testName, 'wb')
    print("Dumping Test Data")
    pickle.dump(testSet, testOut)

    validOut = open(self.validName, 'wb')
    print("Dumping Validation Data")
    pickle.dump(validSet, validOut)