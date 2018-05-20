# [B-31: Imports]
import dataSetGenerator as dsg
import glob
import numpy as np

# keras imports
from keras.models import Sequential
from keras.layers import Dense, Dropout


# [B-32: Data]
# loading the data
inputData = dsg.loadDataset(glob.glob("*_input.txt")[0])
outputData = dsg.loadDataset(glob.glob("*_output.txt")[0])

# reshaping the inputData from (size, col, row) to (size, len), with
# len being col*row
inputDataFlat = inputData.reshape(inputData.shape[0],
                                  inputData.shape[1]*inputData.shape[2])

# reshaping the outputData from (size, col, row) to (size, len)
outputDataFlat = outputData.reshape(outputData.shape[0],
                                    outputData.shape[1]*outputData.shape[2])
# we have now flattend our matrices by concatenating their rows.


# [B-33: Artificial Neural Network]
# neural network
# initialised as Sequential model
model = Sequential()

# [B-33A: add method: Input Layer and Hidden layers]
# input layer
model.add(Dense(units=1000,
                activation="relu",
                input_dim=inputDataFlat.shape[1]))

# dropout layer
model.add(Dropout(rate=0.3))

# hidden layer
model.add(Dense(units=250, activation="relu"))

# output layer
model.add(Dense(units=1, activation="sigmoid"))

# [B-33B: Compile and fit method]
# compilation of the network
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# summary of the network architecture
model.summary()

# start of the training / fitting process
model.fit(x=inputDataFlat,
          y=outputDataFlat,
          epochs=100,
          batch_size=32,
          validation_split=0.2)

# used to keep the console window open after the network is finished training
# so we can read the final accuracy of the network
keepOpen=input("press enter to exit")
