import numpy as np
from datetime import datetime
import os

import eval7, pprint
import copy

import testing_utilities as TUtils
import dataset_utilities as DUtils
import NN_utilities as NNUtils

parameters_ndarray = np.load("weights_2026-01-08 00:01:05@100.0%.npy", allow_pickle=True)
parameters = parameters_ndarray.item()

def nineInputsHighCardPokerDataSetCreation():

    numDict = {"2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
            "7": 6, "8": 7, "9": 8, "T": 9, "J": 10, "Q": 11, "K": 12, "A": 13}

    suitDict = {"s": 1, "h": 2, "c": 3, "d": 4}

    datasetX = []
    datasetY = []
    # num of actual samples is about 6% (3600)
    for i in range(10000000):

        if i % 100000 == 0:
            print( str(i/100000) + "% done")

        deck = eval7.Deck()
        deck.shuffle()
        hand = deck.deal(9)

        hand1 = eval7.evaluate(hand[:7])
        hand1Name = eval7.handtype(hand1)

        hand2 = eval7.evaluate(hand[2:])
        hand2Name = eval7.handtype(hand2)

        if hand1Name == "High Card" and hand2Name == "High Card":

            print(hand)

            baseArray = [0]*9

            for i in range(0,len(hand)):
                cardName = str(hand[i])
                baseArray[i] = numDict[cardName[0]]

            datasetX.append(baseArray)
            datasetY.append(hand1 >= hand2)

            # deep_copied_baseArray = copy.deepcopy(baseArray)
            # deep_copied_baseArray[0], deep_copied_baseArray[1] = deep_copied_baseArray[1], deep_copied_baseArray[0]
            # datasetX.append(deep_copied_baseArray)
            # datasetY.append(hand1 >= hand2)

            # deep_copied_baseArray2 = copy.deepcopy(baseArray)
            # deep_copied_baseArray2[-1], deep_copied_baseArray2[-2] = deep_copied_baseArray2[-2], deep_copied_baseArray2[-1]
            # datasetX.append(deep_copied_baseArray2)
            # datasetY.append(hand1 >= hand2)

            # deep_copied_baseArray3 = copy.deepcopy(baseArray)
            # deep_copied_baseArray3[0], deep_copied_baseArray3[1] = deep_copied_baseArray3[1], deep_copied_baseArray3[0]
            # deep_copied_baseArray3[-1], deep_copied_baseArray3[-2] = deep_copied_baseArray3[-2], deep_copied_baseArray3[-1]
            # datasetX.append(deep_copied_baseArray3)
            # datasetY.append(hand1 >= hand2)

            if len(datasetX) >= 2:
                return datasetX, datasetY

datasetX, datasetY = nineInputsHighCardPokerDataSetCreation()

np.savetxt("9InputsHighCardXTest.csv", datasetX, fmt="%d", delimiter=',')
np.savetxt("9InputsHighCardYTest.csv", datasetY, fmt="%d", delimiter=',')

datasetX = np.loadtxt("9InputsHighCardXTest.csv", delimiter=",", dtype=int)
datasetY = np.loadtxt("9InputsHighCardYTest.csv", delimiter=",", dtype=int)

AL, caches = NNUtils.L_Layer_FeedForward(datasetX.T, parameters)
input()
print(AL)
