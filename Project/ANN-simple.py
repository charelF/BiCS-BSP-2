# [B-31: Imports]
import DatasetGenerator as dsg  # we use the loadDataset function
import numpy as np  # we use the reshape function from numpy
import glob  # a python module to find files matching a specified pattern
# keras imports
from keras.models import Sequential  # we use the Sequential model
from keras.layers import Dense, Dropout
# We explicily import each layer we use to have a more structured code

#-------------------------------------------------------------------------------

# [B-32: Find Dataset]
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

# [B-33 Reshape Dataset]
# reshaping the inputData from (size, col, row) to (size, col*row)
inputDataFlat = inputData.reshape(inputData.shape[0],
                                  inputData.shape[1]*inputData.shape[2])

# reshaping the outputData from (size, col, row) to (size, col*row)
outputDataFlat = outputData.reshape(outputData.shape[0],
                                    outputData.shape[1]*outputData.shape[2])
# we have now flattend our matrices by concatenating their rows.

#-------------------------------------------------------------------------------

# [B-34: Artificial Neural Network]
# neural network
# initialised as Sequential model
model = Sequential()

# [B-34A: adding layers to the network]

# input layer
model.add(Dense(units=1000, activation="relu", input_dim=inputDataFlat.shape[1]))

model.add(Dropout(rate=0.5))  # dropout layer

model.add(Dense(units=250, activation="relu"))  # hidden layer

model.add(Dropout(rate=0.5))  # dropout layer

model.add(Dense(units=25, activation="relu"))  # hidden layer

model.add(Dense(units=1, activation="sigmoid"))  # output layer


# [B-34B: Compiling and fitting the network]

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])  # compiling the network

model.summary()  # summary of the network architecture

# start of the training / fitting process
model.fit(inputDataFlat, outputDataFlat,
          epochs=100, batch_size=32, validation_split=0.2)

# this last method has started the training process, and when we execute it
# and it is finished we will see the accuracy it reached on the dataset

#-------------------------------------------------------------------------------

keepOpen=input()
# used to keep the console window open after the network is finished training
# so we can read the final accuracy of the network
