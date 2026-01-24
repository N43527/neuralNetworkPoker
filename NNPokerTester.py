import testing_utilities as TUtils
import dataset_utilities as DUtils
import NN_utilities as NNUtils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parameters_ndarray = np.load("trainedModels/bestHighCardModel.npy", allow_pickle=True)
parameters = parameters_ndarray.item()

datasetX, datasetY = DUtils.nineInputsHighCardPokerDataSetImport()

trainX, trainY, testX, testY = DUtils.trainTestSplit(datasetX[:300000], datasetY[:300000])

print(str(100*TUtils.predict(trainX, parameters, trainY)) + "%")
print(str(100*TUtils.predict(testX, parameters, testY)) + "%")


outputSampleLength = 4
isDense = True
is18Inputs = False

AL, caches = NNUtils.L_Layer_FeedForward(datasetX.T, parameters)

print("dataset output\n")

isFails = False
TUtils.testOutputPrint(AL, testX, testY, outputSampleLength, isDense, is18Inputs, isFails)

print("\n\ndataset output fails\n")
isFails = True
TUtils.testOutputPrint(AL, trainX, trainY, outputSampleLength, isDense, is18Inputs, isFails)

TUtils.testOutputPrint(AL, testX, testY, outputSampleLength, isDense, is18Inputs, isFails)

progressFile = "datasets/progress.csv"
df = pd.read_csv(progressFile)


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

# Plot each line, specifying the x and y columns and a label for the legend
ax1.plot(df['epoch'], df['cost'], label='cost', marker='o')
ax1.plot(df['epoch'], df['training accuracy'], label='training accuracy', marker='x')
ax1.plot(df['epoch'], df['testing accuracy'], label='testing accuracy', marker='s')

ax1.set_title('Cost, train accuracy, and test accuracy over epochs')
ax1.set_xlabel('num epochs')
ax1.set_ylabel('Cost/Accuracy')
ax1.legend() # Displays the labels defined in plt.plot()
ax1.grid(True) # Adds a grid for better readability

errors, counts = TUtils.predictInDepth(trainX, parameters, trainY)

ax2.bar(errors, counts, color='skyblue')

errors, counts = TUtils.predictInDepth(testX, parameters, testY)

ax3.bar(errors, counts, color='skyblue')

plt.tight_layout()

plt.show()
