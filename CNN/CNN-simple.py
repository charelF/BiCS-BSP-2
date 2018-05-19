# [B-41: Imports]
# personal module imports
import sys
sys.path.append("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
import dataSetGenerator as dsg
# this also imports modules imported in the dataSetGenerator, such as numpy
import glob

# keras imports
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# [B-42: Data]
# loading the data
inputData = dsg.loadDataset(glob.glob("*_input.txt")[0])
outputData = dsg.loadDataset(glob.glob("*_output.txt")[0])
# reshaping the data
inputDataShape = inputData.shape
# reshaping the inputData from (size, col, row) to (size, col, row, channels)
inputData = inputData.reshape(*inputDataShape, 1)
# reshaping the outputData from (size, col, row) to (size, channel)
outputData = outputData.reshape(outputData.shape[0], 1)


# [B-43: Convolutional Neural Network]
# initialised as Sequential model
model = Sequential()

# [B-44A: Convolutional layers]
# input layer
model.add(Conv2D(32, (3, 3), input_shape = (inputDataShape[1], inputDataShape[2], 1), activation = "relu"))

# pooling step
model.add(MaxPooling2D(pool_size = (2, 2)))

# flatten layer
model.add(Flatten())

# [B-44B: ANN]
# hidden layer
model.add(Dense(units=64, activation="relu"))

# output layer
model.add(Dense(units=1, activation="sigmoid"))

# [B-44C: ANN]
# compilation of the network
model.compile(optimizer = "adam",
              loss = "binary_crossentropy",
              metrics = ["accuracy"])

# summary of the network architecture
model.summary()

# start of the training / fitting process
model.fit(x=inputData,
          y=outputData,
          epochs=100,
          batch_size=32,
          validation_split=0.2)

# used to keep the console window open after the network is finished training
# so we can read the final accuracy of the network
keepOpen=input()
