# Imports
import DatasetGenerator as dsg  # we use the loadDataset function
import numpy as np  # we use the reshape function from numpy
import glob  # a python module to find files matching a specified pattern
# keras and scikit imports
from keras.models import Sequential  # we use the Sequential model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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

# reshaping the datasets
inputDataShape = inputData.shape
# reshaping the inputData from (size, col, row) to (size, col, row, channels)
# this does not really change the data, it just changes the dimension
inputData = inputData.reshape(*inputDataShape, 1)
# reshaping the outputData from (size, col, row) to (size, channel)
outputData = outputData.reshape(outputData.shape[0], 1)

#-------------------------------------------------------------------------------

# hyperparameter optimization
# finding the ideal network structure, batch size and epoch amount, values
# activations and optimizer

def neuralNetworkFunction(dropout1, dropout2, dropout3, dense1, dense2, dense3,
                          conv2, pool1, pool2):

    # initialisation and input layer
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     input_shape = (inputDataShape[1], inputDataShape[2], 1),
                     activation = "relu"))


    # using HPO to find the optimal amount and sequence of pooling and
    # convolutional layers
    if pool1:
        model.add(MaxPooling2D(pool_size = (2, 2)))

    if conv2:
        model.add(Conv2D(32, (3, 3), activation="relu"))

    if pool2:
        model.add(MaxPooling2D(pool_size = (2, 2)))


    # flatten layer
    model.add(Flatten())


    # using HPO to find the optimal amount and sequence of dense and dropout layers
    if dropout1:
        model.add(Dropout(rate=0.4))

    if dense1:
        model.add(Dense(units=500, activation="relu"))

    if dropout2:
        model.add(Dropout(rate=0.2))

    if dense2:
        model.add(Dense(units=100, activation="relu"))

    if dropout3:
        model.add(Dropout(rate=0.2))

    if dense3:
        model.add(Dense(units=50, activation="relu"))


    # output layer
    model.add(Dense(units=1, activation="sigmoid"))

    # compilation of the network
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # summary
    model.summary()

    return model


# wrapping the keras neural network into a scikit_learn structure to implement
# hyperparameter optimization
neuralNetworkSKwrap = KerasClassifier(build_fn=neuralNetworkFunction,
                                      epochs=100, batch_size=10)


# definining a parameters dictionary containing all to be tested values for
# each parameter
parameters = {"dropout1":[True, False],
              "dropout2":[True, False],
              "dropout3":[True, False],
              "dense1":[True, False],
              "dense2":[True, False],
              "dense3":[True, False],
              "conv2":[True, False],
              "pool1":[True, False],
              "pool2":[True, False]}


# creating the scikit_learn network from our Keras network and linking
# the parameter dictionary to the network
HPO = GridSearchCV(estimator=neuralNetworkSKwrap,
                   param_grid=parameters,
                   scoring="accuracy",
                   cv=3)  # cv = number of folds

# starting the training process of the final network with the fit method
HPO = HPO.fit(inputData, outputData, validation_split=0.2)
# at this point, the network is being trained

# when the network is finished training, we can read the best configuration
# and the accuracy it reached
bestParameters = HPO.best_params_
bestAccuracy = HPO.best_score_
print(bestParameters, bestAccuracy)

keepOpen=input("press enter to exit")
