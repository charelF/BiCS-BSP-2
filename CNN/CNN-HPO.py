## personal module imports
import sys
sys.path.append("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
import dataSetGenerator as dsg
import glob


# keras imports
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV



# loading the data
inputData = dsg.loadDataset(glob.glob("*_input.txt")[0])
outputData = dsg.loadDataset(glob.glob("*_output.txt")[0])


# reshaping the data
inputDataShape = inputData.shape
# reshaping the inputData from (size, col, row) to (size, col, row, channels)
inputData = inputData.reshape(*inputDataShape, 1)
# reshaping the outputData from (size, col, row) to (size, channel)
outputData = outputData.reshape(outputData.shape[0], 1)


# hyperparameter optimization
# finding the ideal network structure, batch size and epoch amount, values
# activations and optimizer

def neuralNetworkStructure(dropout1, dropout2, dropout3, dropout4,
                           dense1, dense2, dense3, dense4,
                           numNeurons1, numNeurons2, numNeurons3, numNeurons4):



    # initiation of the network
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape = (inputDataShape[1], inputDataShape[2], 1), activation = "relu"))

    # pooling step
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(20, (3, 3), activation="relu"))


    model.add(MaxPooling2D(pool_size = (2, 2)))

    # flatten layer
    model.add(Flatten())


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
structureParameters = {"dropout1":[True, False],
                       "dropout2":[True, False],
                    "dropout3":[True, False],
                    "dropout4":[True, False],
                    "dense1":[True, False],
                    "dense2":[True, False],
                    "dense3":[True, False],
                    "dense4":[True, False],
                    "batch_size":[10, 30, 100, 300],
                    "epochs":[10, 30, 100, 300],
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


"""One idea: create default values for every parameter
    first run the structure parameters (only see the amount of layers)
    secondly run the value parameters (see which number of neurons)
    etc, we continue like this until we have run thtrough al of thems

++  each run takes the best parameters of the previous run as base. Maybe even run twice or more thrhough
    all the arguments until none change anymore.

++  include try + except in function body to prevent whole program shutting down
    when a single configuration failed ( see: my bug I had where the Conv2D
    and MaxPooling2D layer removed too many pixels of the image.)"""
