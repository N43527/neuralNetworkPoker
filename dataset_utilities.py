import numpy as np
import eval7, pprint

folder = "datasets/"

def fullPokerDataSetCreation():
    numDict = {"A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
            "7": 6, "8": 7, "9": 8, "T": 9, "J": 10, "Q": 11, "K": 12}

    suitDict = {"s": 0, "h": 1, "c": 2, "d": 3}

    datasetX = []
    datasetY = []
    for i in range(60000):

        deck = eval7.Deck()
        deck.shuffle()
        hand = deck.deal(9)

        hand1 = eval7.evaluate(hand[:7])
        hand1Name = eval7.handtype(hand1)

        hand2 = eval7.evaluate(hand[2:])
        hand2Name = eval7.handtype(hand2)

        baseArray = [0]*52

        for i in range(len(hand)):
            cardName = str(hand[i])
            cardIndex = numDict[cardName[0]] + suitDict[cardName[1]]*13
            if i < 2:
                baseArray[cardIndex] = 3
            elif i < 7:
                baseArray[cardIndex] = 2
            else:
                baseArray[cardIndex] = 1

        datasetX.append(baseArray)
        datasetY.append(hand1 >= hand2)
    return datasetX, datasetY

def highCardPokerDataSetImport():
    datasetX = np.loadtxt(folder + "highCardX.csv", delimiter=",", dtype=int)[:150000]
    datasetY = np.loadtxt(folder + "highCardY.csv", delimiter=",", dtype=int)[:150000]
    return datasetX, datasetY

def eightteenInputsHighCardPokerDataSetImport():
    datasetX = np.loadtxt(folder + "18InputsHighCardX.csv", delimiter=",")
    datasetY = np.loadtxt(folder + "18InputsHighCardY.csv", delimiter=",", dtype=int)
    return datasetX, datasetY

def eightteenInputsHighCardPokerDataSetHarderImport():
    datasetX = np.loadtxt(folder + "18InputsHighCardX_HarderSamples.csv", delimiter=",")
    datasetY = np.loadtxt(folder + "18InputsHighCardY_HarderSamples.csv", delimiter=",", dtype=int)
    return datasetX, datasetY

def nineInputsHighCardPokerDataSetImport():
    datasetX = np.loadtxt(folder + "9InputsHighCardX.csv", delimiter=",", dtype=int)
    datasetY = np.loadtxt(folder + "9InputsHighCardY.csv", delimiter=",", dtype=int)
    return datasetX, datasetY

def miniDatasetCreation(maxNum, numExamples, minThres):
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