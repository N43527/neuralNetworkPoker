import numpy as np

folder = "datasets/"

def modulo2_Import():
    datasetX = np.loadtxt(folder + "modulo2_X.csv", delimiter=",", dtype=float)
    datasetY = np.loadtxt(folder + "module2_Y.csv", delimiter=",", dtype=int)
    return datasetX, datasetY

def modulo2_extendedInput_Import():
    datasetX = np.loadtxt(folder + "modulo2_X_extendedInput.csv", delimiter=",", dtype=float)
    datasetY = np.loadtxt(folder + "modulo2_Y_extendedInput.csv", delimiter=",", dtype=int)
    return datasetX, datasetY

def x_greater_than_y_times_z_Import():
    datasetX = np.loadtxt(folder + "x_greater_than_y_times_z_X.csv", delimiter=",", dtype=float)
    datasetY = np.loadtxt(folder + "x_greater_than_y_times_z_Y.csv", delimiter=",", dtype=int)
    return datasetX, datasetY

def DatasetCreation(maxNum, numExamples, minThres):
    datasetX = []
    datasetY = []

    datasetX = np.random.rand(numExamples)*maxNum
    datasetX = np.round(datasetX, decimals=2)
    datasetY1 = datasetX > .2
    datasetY2 = datasetX < .4
    datasetY = np.multiply(datasetY1, datasetY2)

    return datasetX, datasetY

def trainTestSplit(datasetX, datasetY):
    datasetX = np.array(datasetX).reshape(len(datasetX), -1)
    datasetY = np.array(datasetY).reshape(len(datasetY), -1)

    cutOff = 7*len(datasetX)//10

    trainX = datasetX[:cutOff].T
    testX = datasetX[cutOff:].T

    trainY = datasetY[:cutOff].T
    testY = datasetY[cutOff:].T

    return trainX, trainY, testX, testY