import numpy as np

epsilon = 1e-19
np.random.seed(1)

def initialize_parameters(layer_dims):
    parameters = {}
    for i in range(1, len(layer_dims)):
        parameters['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) / np.sqrt(layer_dims[i-1])
        parameters["b"+str(i)] = np.zeros((layer_dims[i], 1))

    return parameters

def feedforward(A_prev, W, b, activationType):
    if activationType == "relu":
        Z = np.dot(W, A_prev) + b
        Z_capped = np.maximum(-70, Z)
        A = np.maximum(0, Z_capped)
    elif activationType == "sigmoid":
        Z = np.dot(W, A_prev) + b
        Z_capped = np.maximum(-70, Z)
        A = 1 / (1 + np.exp(-Z_capped))
    
    return [A, ((A_prev, W, b), Z_capped)]

def L_Layer_FeedForward(X, parameters):
    L = len(parameters)//2

    A = X

    caches = []

    for i in range(1, L):
        A_prev = A

        A, cache = feedforward(A_prev, parameters["W"+str(i)], parameters["b"+str(i)], "relu")
        caches.append(cache)
    
    AL, cache = feedforward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")

    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = - (np.dot(Y, np.log(AL+epsilon).T) + np.dot((1-Y), np.log(1-AL+epsilon).T))/m

    return np.squeeze(cost)

def relu_backward(dA, activationCache):
    return np.multiply(dA, activationCache > 0)

def sigmoid_backward(dA, activationCache):

    A = 1 / (1 + np.exp(-activationCache))

    dZ = np.multiply(dA, np.multiply(1-A, A))

    return dZ

def backpropagation(dA, cache, activationType):

    linearCache, activationCache = cache
    A_prev, W, b = linearCache
    m = linearCache[0].shape[1]

    if activationType == "relu":
        dZ = relu_backward(dA, activationCache)
    elif activationType == "sigmoid":
        dZ = sigmoid_backward(dA, activationCache)

    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)

    return [dA_prev, dW, db]

def L_Layer_backpropagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL+epsilon) - np.divide((1-Y), (1-AL+epsilon)))

    dA_prev, dW, db = backpropagation(dAL, caches[L-1], "sigmoid")
    grads["dA_prev" + str(L-1)] = dA_prev
    grads["dW"+str(L)] = dW
    grads["db"+str(L)] = db

    for i in reversed(range(L-1)):
        dA_prev, dW, db = backpropagation(dA_prev, caches[i], "relu")
        grads["dA_prev" + str(i)] = dA_prev
        grads["dW"+str(i+1)] = dW
        grads["db"+str(i+1)] = db
    
    return grads

def updateParameters(parameters, grads, learning_rate):
    L = len(parameters)//2

    for i in range(L):
        parameters["W"+str(i+1)] -= grads["dW"+str(i+1)]*learning_rate
        parameters["b"+str(i+1)] -= grads["db"+str(i+1)]*learning_rate
    
    return parameters

def predict(X, parameters, Y):
    AL, caches = L_Layer_FeedForward(X, parameters)

    ALRound = (AL >= .5).astype(int)

    accuracy = 1 - np.bitwise_xor(ALRound, Y)

    return np.sum(accuracy)/Y.shape[1]