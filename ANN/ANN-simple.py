# personal module imports
import sys
sys.path.append("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
import dataSetGenerator as dsg
import glob


# keras imports
from keras.models import Sequential
from keras.layers import Dense, Dropout


# loading the data
inputData = dsg.loadDataset(glob.glob("*_input.txt")[0])
outputData = dsg.loadDataset(glob.glob("*_output.txt")[0])


# reshaping the 2D matrices (numpy.ndarray) into a 1D list (numpy.ndarray)
# ex: (3000, 32, 32) --> (3000, 1024)
inputDataFlat = inputData.reshape(inputData.shape[0],
                                  inputData.shape[1]*inputData.shape[2])

outputDataFlat = outputData.reshape(outputData.shape[0],
                                    outputData.shape[1]*outputData.shape[2])

# neural network
# initialised as Sequential model (necessary)
model = Sequential()

# input layer (necessary)
model.add(Dense(units=1000, activation='relu', input_dim=inputDataFlat.shape[1]))

# hidden layer (optional)
model.add(Dense(units=250, activation='relu'))

# output layer (necessary)
model.add(Dense(units=1, activation='sigmoid'))

# compilation of the network (necessary)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# summary of the network architecture (optional)
model.summary()

# start of the training / fitting process (necessary)
model.fit(inputDataFlat,
          outputDataFlat,
          epochs=30,
          validation_split=0.2)

# used to keep the console window open after the network is finished training
# so we can read the metrics
keepOpen=input("press enter to exit")
