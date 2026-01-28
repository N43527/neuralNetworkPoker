# Overall flow
# 1. initial parameters
# 2. feedforward
# 3. compute cost
# 4. backward propagation
# 5. update weights

import numpy as np
from datetime import datetime
import os

import NN_utilities as NNUtils
import dataset_utilities as DUtils

start_time = datetime.now()
timeStamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
print(f"Start Time: {start_time:%H:%M:%S}")

# Need to change
datasetX, datasetY = DUtils.modulo2_extendedInput_Import()

trainX, trainY, testX, testY = DUtils.trainTestSplit(datasetX, datasetY)

learning_rate = 0.015
layer_dims = [trainX.shape[0], 20, 20, 20, 20, trainY.shape[0]]
outputSampleLength = 4
epochs = 200000

trainedModelFolder = "trainedModels/"
datasetsFolder = "datasets/"
progressFile = datasetsFolder + "progress.csv"
with open(progressFile, "w") as f:
    f.write("epoch,cost,training accuracy,testing accuracy\n")
progressList = []
startCounter = 0

prevName = ""

# Step 1: initial parameters

usePrev = ""
if usePrev != "":
    parameters_ndarray = np.load(usePrev, allow_pickle=True)
    parameters = parameters_ndarray.item()
else:
    parameters = NNUtils.initialize_parameters(layer_dims)

for i in range(epochs):

    # Step 2: feedforward
    AL, caches = NNUtils.L_Layer_FeedForward(trainX, parameters)

    # Step 3: compute cost
    cost = NNUtils.compute_cost(AL, trainY)
    trainAccuracy = NNUtils.predict(trainX, parameters, trainY)
    testAccuracy = NNUtils.predict(testX, parameters, testY)

    progressList.append([cost, trainAccuracy, testAccuracy])

    if i % 1000 == 0:
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"{str(i*100/epochs)}% done after {str(duration).split('.')[0]} (H:M:S)")
        print("cost of iteration #", i, "is", cost)
        print("train accuracy: " + str(100*trainAccuracy) + "%")
        print("test accuracy: "+ str(100*testAccuracy) + "%")
        filename = trainedModelFolder + "weights_" + timeStamp + "@"+ str(i*100/epochs) + "%.npy"
        if prevName != "":
            os.replace(prevName, filename)
        np.save(filename, parameters)
        prevName = filename

        with open(progressFile, "a") as f:
            for cost, trainAccuracy, testAccuracy in progressList:
                f.write(f"{startCounter},{cost},{trainAccuracy},{testAccuracy}\n")
                startCounter += 1
            progressList = []


    # Step 4: back propagate
    grads = NNUtils.L_Layer_backpropagation(AL, trainY, caches)

    # Step 5: update parameters
    parameters = NNUtils.updateParameters(parameters, grads, learning_rate)

with open(progressFile, "a") as f:
    for cost, trainAccuracy, testAccuracy in progressList:
        f.write(f"{startCounter},{cost},{trainAccuracy},{testAccuracy}\n")
        startCounter += 1
    progressList = []

end_time = datetime.now()
print(f"End Time:   {end_time:%H:%M:%S}")

duration = end_time - start_time
print(f"Total Duration:   {str(duration).split('.')[0]} (H:M:S)")


print("train accuracy: ", str(100*NNUtils.predict(trainX, parameters, trainY)) + "%")
print("test accuracy: ", str(100*NNUtils.predict(testX, parameters, testY)) + "%")

filename = trainedModelFolder + "weights_" + timeStamp + "@"+ str((i+1)*100/epochs) + "%.npy"
if prevName != "":
    os.replace(prevName, filename)
np.save(filename, parameters)

print(filename)
