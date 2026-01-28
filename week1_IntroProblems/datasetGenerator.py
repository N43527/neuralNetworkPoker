import numpy as np

def modulo2DataSetCreation():

    datasetX = []
    datasetY = []
    numDataPoints = 100000
    for i in range(numDataPoints):

        if i % (numDataPoints//100) == 0:
            print(str(100*i/(numDataPoints)) + "% done")

        num = np.random.randint(0, (i%1000+10)/10)
        datasetX.append(num/10)
        datasetY.append(num%2 == 0)

    return datasetX, datasetY

def modulo2_extendedInput_DataSetCreation():

    datasetX = []
    datasetY = []
    numDataPoints = 100000
    for i in range(numDataPoints):

        if i % (numDataPoints//100) == 0:
            print(str(100*i/(numDataPoints)) + "% done")

        num = np.random.randint(0, (i%1000+10)/10)
        datasetX.append([num/10, num%2 == 0])
        datasetY.append(num%2 == 0)

    return datasetX, datasetY

def x_greater_than_y_times_z():

    datasetX = []
    datasetY = []
    numDataPoints = 100000
    for i in range(numDataPoints):

        if i % (numDataPoints//100) == 0:
            print(str(100*i/(numDataPoints)) + "% done")

        x = np.random.rand()*2.5
        y = np.random.rand()
        z = np.random.randint(2,4)
        datasetX.append([x, y, z])
        datasetY.append(x > y*z)

    return datasetX, datasetY


datasetsFolder = "datasets/"

# X, Y = modulo2DataSetCreation()
# print(len(X))

# np.savetxt(datasetsFolder + "modulo2_X.csv", X, fmt="%2f", delimiter=',')
# np.savetxt(datasetsFolder + "module2_Y.csv", Y, fmt="%d", delimiter=',')

# X, Y = modulo2_extendedInput_DataSetCreation()
# print(len(X))

# np.savetxt(datasetsFolder + "modulo2_X_extendedInput.csv", X, fmt="%2f", delimiter=',')
# np.savetxt(datasetsFolder + "modulo2_Y_extendedInput.csv", Y, fmt="%d", delimiter=',')

X, Y = x_greater_than_y_times_z()
print(len(X))

np.savetxt(datasetsFolder + "x_greater_than_y_times_z_X.csv", X, fmt="%2f", delimiter=',')
np.savetxt(datasetsFolder + "x_greater_than_y_times_z_Y.csv", Y, fmt="%d", delimiter=',')

