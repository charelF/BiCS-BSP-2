# personal module imports
import DatasetGenerator as dsg
import numpy as np
import glob


# keras imports
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


# loading the data
try:
    inputData = dsg.loadDataset(glob.glob("*_input.txt")[0])
    outputData = dsg.loadDataset(glob.glob("*_output.txt")[0])
except IndexError:
    try:
        import os
        os.chdir(os.path.realpath("Project"))
        inputData = dsg.loadDataset(glob.glob("*_input.txt")[0])
        outputData = dsg.loadDataset(glob.glob("*_output.txt")[0])
    except IndexError:
        print("there are probably no datasets in the project folder")


# reshaping the 2D matrices (numpy.ndarray) into a 1D list (numpy.ndarray)
# reshaping the inputData from (size, col, row) to (size, len), with
# len being col*row and thus the lenght of the list
inputDataFlat = inputData.reshape(inputData.shape[0],
                                  inputData.shape[1]*inputData.shape[2])

# reshaping the outputData from (size, col, row) to (size, len)
outputDataFlat = outputData.reshape(outputData.shape[0],
                                    outputData.shape[1]*outputData.shape[2])



# hyperparameter optimization
# finding the ideal network values, activation functions, neurons per layer, ...

# creating a function containing the neural network architecture
def neuralNetworkStructure(dropoutRate1, dropoutRate2, activationInput,
                           activation, activationOutput, kernelInit, optimizer):

    # print arguments
    print("in this run, the following arguments are used:\n",
          dropoutRate1, dropoutRate2, activation, activationOutput,
          activationInput, kernelInit, optimizer)

    # initiation of the network
    model = Sequential()

    # input layer
    model.add(Dense(units=1000,
                    activation=activationInput,
                    input_dim=inputData.shape[1]*inputData.shape[2],
                    kernel_initializer=kernelInit))

    # hidden layers
    model.add(Dropout(rate=dropoutRate1))

    model.add(Dense(units=500, kernel_initializer=kernelInit, activation=activation))

    model.add(Dropout(rate=dropoutRate2))

    model.add(Dense(units=50, kernel_initializer=kernelInit, activation=activation))

    # output layer
    model.add(Dense(units=1, activation=activationOutput))

    # compilation of the network
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # summary
    model.summary()

    return model


# wrapping the keras neural network into a scikit_learn structure to implement
# hyperparameter optimization
neuralNetwork = KerasClassifier(build_fn=neuralNetworkStructure,
                                epochs=100,
                                batch_size=32)


# definining a parameters dictionary containing all to be tested arguments for
# each parameter
parameters = {"dropoutRate1":[0.0, 0.1, 0.2, 0.5, 0.7],
              "dropoutRate2":[0.0, 0.1, 0.2, 0.5, 0.7],
              "activationInput":["relu", "tanh", "linear"],
              "activation":["sigmoid", "relu", "softmax", "tanh", "linear"],
              "activationOutput":["sigmoid", "softmax", "tanh", "linear"],
              "kernelInit":["uniform", "normal", "glorot_uniform", "he_uniform"],
              "optimizer":['SGD', 'RMSprop', 'adam']}



# creating the scikit_learn network from our Keras network and linking
# the parameter dictionary to the network
gridSearch = GridSearchCV(estimator=neuralNetwork,
                          param_grid=parameters,
                          scoring="accuracy",
                          cv=3)  # cv = number of folds


# starting the training process of the final network with the fit method
gridSearch = gridSearch.fit(inputDataFlat, outputDataFlat, validation_split=0.2)


# at this point, the network is being trained


# when the network is finished training, we can read the best configuration
# and the accuracy it reached
bestParameters = gridSearch.best_params_
bestAccuracy = gridSearch.best_score_
print(bestParameters, bestAccuracy)

keepOpen=input("press enter to exit")
