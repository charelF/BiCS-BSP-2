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
inputData = dsg.loadDataset(glob.glob("*_input.txt")[0])
outputData = dsg.loadDataset(glob.glob("*_output.txt")[0])


# reshaping the 2D matrices (numpy.ndarray) into a 1D list (numpy.ndarray)
# reshaping the inputData from (size, col, row) to (size, len), with
# len being col*row and thus the lenght of the list
inputDataFlat = inputData.reshape(inputData.shape[0],
                                  inputData.shape[1]*inputData.shape[2])

# reshaping the outputData from (size, col, row) to (size, len)
outputDataFlat = outputData.reshape(outputData.shape[0],
                                    outputData.shape[1]*outputData.shape[2])


# hyperparameter optimization
# finding the ideal network structure, batch size and epoch amount

# creating a function containing the neural network architecture
def neuralNetworkStructure(dropout1, dropout2, dropout3, dropout4,
                           dense1, dense2, dense3, dense4,
                           numNeurons1, numNeurons2, numNeurons3, numNeurons4):

    # print arguments
    print("in this run, the following arguments are used:\n",
          dropout1, dropout2, dropout3, dropout4,
          dense1, dense2, dense3, dense4,
          numNeurons1, numNeurons2, numNeurons3, numNeurons4)

    # initiation of the network
    model = Sequential()

    # input layer
    model.add(Dense(units=numNeurons0,
                    activation="relu",
                    input_dim=inputData.shape[1]*inputData.shape[2]))

    # optional hidden layers
    if dropout1:
        model.add(Dropout(rate=0.4))
    if dense1:
        model.add(Dense(units=numNeurons1, activation="relu"))
    if dropout2:
        model.add(Dropout(rate=0.2))
    if dense2:
        model.add(Dense(units=numNeurons2, activation="relu"))
    if dropout3:
        model.add(Dropout(rate=0.2))
    if dense3:
        model.add(Dense(units=numNeurons3, activation="relu"))
    if dropout4:
        model.add(Dropout(rate=0.2))
    if dense4:
        model.add(Dense(units=numNeurons4, activation="relu"))

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
neuralNetwork = KerasClassifier(build_fn=neuralNetworkStructure)


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
              "numNeurons4":[25, 10, 5]}


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
