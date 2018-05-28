# [B-41: Imports]
import DatasetGenerator as dsg  # we use the loadDataset function
import numpy as np  # we use the reshape function from numpy
import glob  # a python module to find files matching a specified pattern
# keras imports
from keras.models import Sequential  # we use the Sequential model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# We explicily import each layer we use to have a more structured code

#-------------------------------------------------------------------------------

# [B-42: Find Dataset]
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

# [B-43 Reshape Dataset]
inputDataShape = inputData.shape
# reshaping the inputData from (size, col, row) to (size, col, row, channels)
# this does not really change the data, it just changes the dimension
inputData = inputData.reshape(*inputDataShape, 1)
# reshaping the outputData from (size, col, row) to (size, channel)
outputData = outputData.reshape(outputData.shape[0], 1)

#-------------------------------------------------------------------------------

# [B-44: Convolutional Neural Network]
# initialised as Sequential model
model = Sequential()

# [B-44A: adding Convolutional and pooling layers]

# input layer
model.add(Conv2D(filters=16, kernel_size=(3, 3),
                 input_shape = (inputDataShape[1], inputDataShape[2], 1),
                 activation = "relu"))
# input shape (col, row, channel) -->
#                    (col-(kernel_size[0]-1), row-(kernel_size[0]-1), filters)
# ex: (64, 64, 1)  --> (62, 62, 16)

model.add(MaxPooling2D(pool_size = (2, 2)))  # pooling step
# input shape (col, row, filters) --> (col/2), row-/2), filters)
# ex: (62, 62, 16) --> (31, 31, 16)

model.add(Conv2D(filters=8, kernel_size=(3, 3), activation = "relu"))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten()) # flatten layer
# input shape (col, row, filters) --> output shape (col*row*filters)
# ex: (31, 31, 32) --> (30752)

model.add(Dropout(rate=0.6))
# [B-44B: adding ANN layers such as Dense]

model.add(Dense(units=256, activation="relu"))  # hidden layer

model.add(Dropout(rate=0.5))

model.add(Dense(units=32, activation="relu"))

model.add(Dropout(rate=0.4))

model.add(Dense(units=1, activation="sigmoid"))  # output layer


# [B-44C: Compiling and fitting the network]

model.compile(optimizer = "adam",
              loss = "binary_crossentropy",
              metrics = ["accuracy"])  # compiling the network


model.summary()  # summary of the network architecture

# start of the training / fitting process
model.fit(inputData, outputData, epochs=100, batch_size=32, validation_split=0.2)

# this last method has started the training process, and when we execute it
# and it is finished we will see the accuracy it reached on the dataset

#-------------------------------------------------------------------------------

keepOpen=input()
# used to keep the console window open after the network is finished training
# so we can read the final accuracy of the network

# ----------------- unused, but sometimes needed while debugging ---------------

# import sys
# import os
# sys.path.append(os.path.realpath(".."))
# sys.path.append(os.path.realpath("\\Dataset_Generator"))
# sys.path.append(os.path.realpath("/Dataset_Generator"))
# import os
# os.chdir(os.path.realpath("Project"))
#
# from Dataset_Generator import dataSetGenerator as dsg
