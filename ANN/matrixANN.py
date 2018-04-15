# personal module imports

import sys
sys.path.append("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
import dataSetGenerator as dsg


# keras imports

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# data

inputData = dsg.loadDataset("test_input.txt")
outputData = dsg.loadDataset("test_output.txt")
print("the size of the dataset is: ", inputData.shape, " of type: ", type(inputData))

inputDataFlat = inputData.reshape(inputData.shape[0], inputData.shape[1]*inputData.shape[2])
outputDataFlat = outputData.reshape(outputData.shape[0], outputData.shape[1]*outputData.shape[2])


# ANN

ann = Sequential()

ann.add(Dense(units=1000, activation='relu', input_dim=inputData.shape[1]*inputData.shape[2]))  # input

ann.add(Dropout(rate=0.4))

ann.add(Dense(units=400, activation='relu'))

ann.add(Dropout(rate=0.4))

ann.add(Dense(units=40, activation='relu'))  # hidden

ann.add(Dropout(rate=0.4))

ann.add(Dense(units=1, activation='sigmoid'))  # output

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.summary()

ann.fit(inputDataFlat,
        outputDataFlat,
        epochs=25,
        validation_split=0.2)

keepOpen=input()

# Summary: This network is completely working and can be tuned, modified an
# studied to understand its behvaiour.
