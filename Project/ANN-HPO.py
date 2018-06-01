# Imports
import DatasetGenerator as dsg  # we use the loadDataset function
import numpy as np  # we use the reshape function from numpy
import glob  # a python module to find files matching a specified pattern
# keras and scikit imports
from keras.models import Sequential  # we use the Sequential model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
# We explicily import each layer we use to have a more structured code

#-------------------------------------------------------------------------------

# loading the data
try:
    # we use a try statememt as this section is prone to failling if
    # no dataset has been created or can be found.
    inputData = dsg.loadDataset(glob.glob("*_input.txt")[0])
    outputData = dsg.loadDataset(glob.glob("*_output.txt")[0])
    # we search for every file in our filepath that ends in "_input.txt"
    # respectively "_output.txt"
except IndexError:
    try:
        # first error source: in my case, sometimes the python script searched
        # in the wrong directories
        import os
        os.chdir(os.path.realpath("Project"))
        inputData = dsg.loadDataset(glob.glob("*_input.txt")[0])
        outputData = dsg.loadDataset(glob.glob("*_output.txt")[0])
    except IndexError:  # second error source
        print("there are probably no datasets in the project folder")

#-------------------------------------------------------------------------------

# reshaping the inputData from (size, col, row) to (size, col*row)
inputDataFlat = inputData.reshape(inputData.shape[0],
                                  inputData.shape[1]*inputData.shape[2])

# reshaping the outputData from (size, col, row) to (size, col*row)
outputDataFlat = outputData.reshape(outputData.shape[0],
                                    outputData.shape[1]*outputData.shape[2])
# we have now flattend our matrices by concatenating their rows.

#-------------------------------------------------------------------------------

# hyperparameter optimization
# finding the ideal network structure, batch size and epoch amount

# creating a function containing the neural network architecture
def neuralNetworkFunction(dropout1=True, dropout2=True, dropout3=False,dropout4=False,
          dense1=True, dense2=True, dense3=False, dense4=False, numNeurons0=2000,
          numNeurons1=1000, numNeurons2=250, numNeurons3=50, numNeurons4 = 10,
          dropoutRate1=0.2, dropoutRate2=0.2, dropoutRate3=0.2, dropoutRate4=0.2,
          activationInput="relu", activation="relu", activationOutput="sigmoid",
          kernelInit="uniform", optimizer="adam"):



    # initiation of the network
    model = Sequential()

    # input layer
    model.add(Dense(units=numNeurons0,
                    activation=activationInput,
                    input_dim=inputData.shape[1]*inputData.shape[2]))

    # optional hidden layers
    if dropout1:
        model.add(Dropout(rate=dropoutRate1))
    if dense1:
        model.add(Dense(units=numNeurons1, activation=activation))
    if dropout2:
        model.add(Dropout(rate=dropoutRate2))
    if dense2:
        model.add(Dense(units=numNeurons2, activation=activation))
    if dropout3:
        model.add(Dropout(rate=dropoutRate3))
    if dense3:
        model.add(Dense(units=numNeurons3, activation=activation))
    if dropout4:
        model.add(Dropout(rate=dropoutRate4))
    if dense4:
        model.add(Dense(units=numNeurons4, activation=activation))

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
neuralNetworkSKwrap = KerasClassifier(build_fn=neuralNetworkFunction)


# definining a parameters dictionary containing all to be tested values for
# each parameter
parameters = {"dropout1":[True, False],
              "dropout2":[True, False],
              "dropout3":[True, False],
              "dropout4":[True, False],
              "dense1":[True, False],
              "dense2":[True, False],
              "dense3":[True, False],
              "dense4":[True, False],
              "batch_size":[10, 30, 100, 300],
              "epochs":[10, 30, 100, 300],
              "numNeurons0":[2000, 1000, 500],
              "numNeurons1":[2000, 1000, 500],
              "numNeurons2":[500, 250, 100],
              "numNeurons3":[100, 50, 25],
              "numNeurons4":[25, 10, 5],
              "dropoutRate1":[0.0, 0.1, 0.2, 0.5, 0.7],
              "dropoutRate2":[0.0, 0.1, 0.2, 0.5, 0.7],
              "activationInput":["relu", "tanh", "linear"],
              "activation":["sigmoid", "relu", "softmax", "tanh", "linear"],
              "activationOutput":["sigmoid", "softmax", "tanh", "linear"],
              "kernelInit":["uniform", "normal", "glorot_uniform", "he_uniform"],
              "optimizer":['SGD', 'RMSprop', 'adam']}


parameters1 = {"dropout1":[True, False],
               "dropout2":[True, False],
               "dropout3":[True, False],
               "dropout4":[True, False]}

parameters2 = {"dense1":[True, False],
               "dense2":[True, False],
               "dense3":[True, False],
               "dense4":[True, False]}

parameters3 = {"batch_size":[10, 30, 100, 300],
               "epochs":[10, 30, 100, 300]}

parameters4 = {"numNeurons0":[2000, 1000, 500],
               "numNeurons1":[2000, 1000, 500]}

parameters5 = {"numNeurons2":[500, 250, 100],
               "numNeurons3":[100, 50, 25],
               "numNeurons4":[25, 10, 5]}

parameters6 = {"dropoutRate1":[0.0, 0.1, 0.2, 0.5, 0.7],
               "dropoutRate2":[0.0, 0.1, 0.2, 0.5, 0.7]}

parameters7 = {"activationInput":["relu", "tanh", "linear"],
               "activation":["sigmoid", "relu", "softmax", "tanh", "linear"]}

parameters8 = {"activationOutput":["sigmoid", "softmax", "tanh", "linear"],
               "kernelInit":["uniform", "normal", "glorot_uniform", "he_uniform"]}

parameters9 = {"optimizer":['SGD', 'RMSprop', 'adam']}



HPO = GridSearchCV(estimator=neuralNetworkSKwrap,
                   param_grid=parameters1,
                   scoring="accuracy",
                   cv=3)  # cv = number of folds

HPO = HPO.fit(inputDataFlat, outputDataFlat, validation_split=0.2)

bestParameters = HPO.best_params_
bestAccuracy = HPO.best_score_
print(bestParameters, bestAccuracy)

keepOpen=input("press enter to exit")
