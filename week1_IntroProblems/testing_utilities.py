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

def testOutputPrint_XGreaterYTimesZ(AL, X, Y, numExamples, isFails):
    
    intY = Y.astype(int)
    ALRound = np.round(AL, decimals=2)

    if isFails:
        count = 0
        i = 0
        while count < numExamples and i < X.shape[1]:
            example = X[:, [i]].reshape(-1)

            if (intY[0][i] == 1 and AL[0][i] < .5) or  (intY[0][i] == 0 and AL[0][i] >= .5):
                print("example is " + str(example))
                print("prediction is", str(ALRound[0][i]))
                print("actual is", str(intY[0][i]))
                count += 1
            i += 1
    else:
        for i in range(numExamples):
            example = X[:, [i]].reshape(-1)

            print("example is " + str(example))
            print("prediction is", str(ALRound[0][i]))
            print("actual is", str(intY[0][i]))
    