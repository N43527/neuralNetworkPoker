import numpy as np
import eval7, pprint
import copy

numDict = {"A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
           "7": 6, "8": 7, "9": 8, "T": 9, "J": 10, "Q": 11, "K": 12}

suitDict = {"s": 0, "h": 1, "c": 2, "d": 3}

datasetX = []
datasetY = []
for i in range(10000):

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

datasetX = np.array(datasetX).T

datasetY = np.array(datasetY).reshape(len(datasetY), -1).T


def highCardPokerDataSetCreation():
    import numpy as np
    import eval7, pprint

    numDict = {"A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
            "7": 6, "8": 7, "9": 8, "T": 9, "J": 10, "Q": 11, "K": 12}

    suitDict = {"s": 0, "h": 1, "c": 2, "d": 3}

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

def eighteenInputsHighCardPokerDataSetCreation():

    numDict = {"2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
            "7": 6, "8": 7, "9": 8, "T": 9, "J": 10, "Q": 11, "K": 12, "A": 13}

    suitDict = {"s": 1, "h": 2, "c": 3, "d": 4}

    datasetX = []
    datasetY = []
    # num of actual samples is about 6% (3600)
    for i in range(1000000):

        if i % 10000 == 0:
            print( str(i/10000) + "% done")

        deck = eval7.Deck()
        deck.shuffle()
        hand = deck.deal(9)

        hand1 = eval7.evaluate(hand[:7])
        hand1Name = eval7.handtype(hand1)

        hand2 = eval7.evaluate(hand[2:])
        hand2Name = eval7.handtype(hand2)

        if hand1Name == "High Card" and hand2Name == "High Card":

            baseArray = [0.0]*18

            for i in range(0,len(hand)):
                cardName = str(hand[i])
                baseArray[2*i] = numDict[cardName[0]]/16
                baseArray[2*i+1] = suitDict[cardName[1]]/4

            if str(hand1)[0] == str(hand2)[0] or (hand1 < 800000 and hand2 < 800000):
                datasetX.append(baseArray)
                datasetY.append(hand1 >= hand2)

                deep_copied_baseArray = copy.deepcopy(baseArray)
                deep_copied_baseArray[0], deep_copied_baseArray[2] = deep_copied_baseArray[2], deep_copied_baseArray[0]
                deep_copied_baseArray[1], deep_copied_baseArray[3] = deep_copied_baseArray[3], deep_copied_baseArray[1]
                datasetX.append(deep_copied_baseArray)
                datasetY.append(hand1 >= hand2)

                deep_copied_baseArray2 = copy.deepcopy(baseArray)
                deep_copied_baseArray2[-1], deep_copied_baseArray2[-3] = deep_copied_baseArray2[-3], deep_copied_baseArray2[-1]
                deep_copied_baseArray2[-2], deep_copied_baseArray2[-4] = deep_copied_baseArray2[-4], deep_copied_baseArray2[-2]
                datasetX.append(deep_copied_baseArray2)
                datasetY.append(hand1 >= hand2)

                deep_copied_baseArray3 = copy.deepcopy(baseArray)
                deep_copied_baseArray3[0], deep_copied_baseArray3[2] = deep_copied_baseArray3[2], deep_copied_baseArray3[0]
                deep_copied_baseArray3[1], deep_copied_baseArray3[3] = deep_copied_baseArray3[3], deep_copied_baseArray3[1]
                deep_copied_baseArray3[-1], deep_copied_baseArray3[-3] = deep_copied_baseArray3[-3], deep_copied_baseArray3[-1]
                deep_copied_baseArray3[-2], deep_copied_baseArray3[-4] = deep_copied_baseArray3[-4], deep_copied_baseArray3[-2]
                datasetX.append(deep_copied_baseArray3)
                datasetY.append(hand1 >= hand2)

    return datasetX, datasetY

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

            baseArray = [0]*9

            for i in range(0,len(hand)):
                cardName = str(hand[i])
                baseArray[i] = numDict[cardName[0]]

            datasetX.append(baseArray)
            datasetY.append(hand1 >= hand2)

            deep_copied_baseArray = copy.deepcopy(baseArray)
            deep_copied_baseArray[0], deep_copied_baseArray[1] = deep_copied_baseArray[1], deep_copied_baseArray[0]
            datasetX.append(deep_copied_baseArray)
            datasetY.append(hand1 >= hand2)

            deep_copied_baseArray2 = copy.deepcopy(baseArray)
            deep_copied_baseArray2[-1], deep_copied_baseArray2[-2] = deep_copied_baseArray2[-2], deep_copied_baseArray2[-1]
            datasetX.append(deep_copied_baseArray2)
            datasetY.append(hand1 >= hand2)

            deep_copied_baseArray3 = copy.deepcopy(baseArray)
            deep_copied_baseArray3[0], deep_copied_baseArray3[1] = deep_copied_baseArray3[1], deep_copied_baseArray3[0]
            deep_copied_baseArray3[-1], deep_copied_baseArray3[-2] = deep_copied_baseArray3[-2], deep_copied_baseArray3[-1]
            datasetX.append(deep_copied_baseArray3)
            datasetY.append(hand1 >= hand2)

    return datasetX, datasetY


# X, Y = highCardPokerDataSetCreation()
# print(len(X))

# np.savetxt("highCardX.csv", X, fmt="%d", delimiter=',')
# np.savetxt("highCardY.csv", Y, fmt="%d", delimiter=',')

# X, Y = eighteenInputsHighCardPokerDataSetCreation()
# print(len(X))

# np.savetxt("18InputsHighCardX_HarderSamples.csv", X, fmt="%1f", delimiter=',')
# np.savetxt("18InputsHighCardY_HarderSamples.csv", Y, fmt="%d", delimiter=',')

X, Y = nineInputsHighCardPokerDataSetCreation()
print(len(X))

np.savetxt("9InputsHighCardX.csv", X, fmt="%d", delimiter=',')
np.savetxt("9InputsHighCardY.csv", Y, fmt="%d", delimiter=',')


