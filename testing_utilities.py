import numpy as np
import NN_utilities as NNUtils
from collections import defaultdict



def predict(X, parameters, Y):
    AL, caches = NNUtils.L_Layer_FeedForward(X, parameters)

    ALRound = (AL >= .5).astype(int)

    accuracy = 1 - np.bitwise_xor(ALRound, Y)

    return np.sum(accuracy)/Y.shape[1]

def predictInDepth(X, parameters, Y):
    AL, caches = NNUtils.L_Layer_FeedForward(X, parameters)

    counts = defaultdict(int) 

    ALDiff = np.round(np.abs(AL-Y), decimals=2).T

    print(ALDiff.shape)

    for a in ALDiff:
        counts[a[0]] += 1

    return counts.keys(), counts.values()

def arrayToHandConverter52(xArray):
    myHand = []
    communityHand = []
    oppHand = []

    reverseNumDict = {0:"A", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6",
           6: "7", 7: "8", 8: "9", 9: "T", 10: "J", 11: "Q", 12: "K"}

    reverseSuitDict = {0: "s", 1: "h", 2: "c", 3: "d"}

    for i in range(len(xArray)):
        cardName = reverseNumDict[i%13]+reverseSuitDict[i//13]
        if xArray[i] == 3:
            myHand.append(cardName)
        elif xArray[i] == 2:
            communityHand.append(cardName)
        elif xArray[i] == 1:
            oppHand.append(cardName)
    
    return myHand, communityHand, oppHand

def arrayToHandConverter18(xArray):
    myHand = []
    communityHand = []
    oppHand = []

    eighteenInputsReverseNumDict = {1: "2", 2: "3", 3: "4", 4: "5", 5: "6",
           6: "7", 7: "8", 8: "9", 9: "T", 10: "J", 11: "Q", 12: "K", 13:"A",}

    eighteenInputsReverseSuitDict = {1: "s", 2: "h", 3: "c", 4: "d"}

    for i in range(len(xArray)//2):
        # print(int(xArray[2*i]*16))
        # print(int(xArray[2*i+1]*4))
        # print("next card")
        cardName = eighteenInputsReverseNumDict[int(xArray[2*i]*16)] + eighteenInputsReverseSuitDict[int(xArray[2*i+1]*4)]
        if i < 2:
            myHand.append(cardName)
        elif i < 7:
            communityHand.append(cardName)
        else:
            oppHand.append(cardName)
    
    return myHand, communityHand, oppHand

def arrayToHandConverter9(xArray):
    myHand = []
    communityHand = []
    oppHand = []

    nineInputsReverseNumDict = {1: "2", 2: "3", 3: "4", 4: "5", 5: "6",
           6: "7", 7: "8", 8: "9", 9: "T", 10: "J", 11: "Q", 12: "K", 13:"A",}

    for i in range(len(xArray)):
        cardName = nineInputsReverseNumDict[int(xArray[i])]
        if i < 2:
            myHand.append(cardName)
        elif i < 7:
            communityHand.append(cardName)
        else:
            oppHand.append(cardName)
    
    return myHand, communityHand, oppHand

def testOutputPrint(AL, X, Y, numExamples, isDense, is18Inputs, isFails):
    
    intY = Y.astype(int)
    ALRound = np.round(AL, decimals=2)

    if isFails:
        count = 0
        i = 0
        while count < numExamples and i < X.shape[1]:
            example = X[:, [i]].reshape(-1)
            if isDense:
                if is18Inputs:
                    myHand, communityHand, oppHand = arrayToHandConverter18(example)
                else:
                    myHand, communityHand, oppHand = arrayToHandConverter9(example)
            else:
                myHand, communityHand, oppHand = arrayToHandConverter52(example)

            if (intY[0][i] == 1 and AL[0][i] < .5) or  (intY[0][i] == 0 and AL[0][i] >= .5):
                print("myHand is", myHand[0], "&", myHand[1])
                print("communityHand is", communityHand[0], "&", communityHand[1], "&", communityHand[2], "&", communityHand[3], "&", communityHand[4])
                print("oppHand is", oppHand[0], "&", oppHand[1])
                print(ALRound[0][i])
                print(intY[0][i])
                count += 1
            i += 1
    else:
        for i in range(numExamples):
            example = X[:, [i]].reshape(-1)

            if isDense:
                if is18Inputs:
                    myHand, communityHand, oppHand = arrayToHandConverter18(example)
                else:
                    myHand, communityHand, oppHand = arrayToHandConverter9(example)
            else:
                myHand, communityHand, oppHand = arrayToHandConverter52(example)

            print("myHand is", myHand[0], "&", myHand[1])
            print("communityHand is", communityHand[0], "&", communityHand[1], "&", communityHand[2], "&", communityHand[3], "&", communityHand[4])
            print("oppHand is", oppHand[0], "&", oppHand[1])
            print(ALRound[0][i])
            print(intY[0][i])
    