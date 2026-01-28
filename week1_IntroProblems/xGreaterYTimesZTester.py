import testing_utilities as TUtils
import dataset_utilities as DUtils
import NN_utilities as NNUtils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parameters_ndarray = np.load("trainedModels/weights_2026-01-28 01:37:56@100.0%.npy", allow_pickle=True)
parameters = parameters_ndarray.item()

datasetX, datasetY = DUtils.x_greater_than_y_times_z_Import()

trainX, trainY, testX, testY = DUtils.trainTestSplit(datasetX, datasetY)

print(str(100*TUtils.predict(trainX, parameters, trainY)) + "%")
print(str(100*TUtils.predict(testX, parameters, testY)) + "%")


outputSampleLength = 10

AL, caches = NNUtils.L_Layer_FeedForward(datasetX.T, parameters)

print("dataset output\n")
isFails = False
TUtils.testOutputPrint_XGreaterYTimesZ(AL, testX, testY, outputSampleLength, isFails)

print("\n\ndataset output fails\n")
isFails = True
TUtils.testOutputPrint_XGreaterYTimesZ(AL, trainX, trainY, outputSampleLength, isFails)
print("\n\ntest output fails\n")
TUtils.testOutputPrint_XGreaterYTimesZ(AL, testX, testY, outputSampleLength, isFails)

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
