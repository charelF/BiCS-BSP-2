# personal module imports
import sys
sys.path.append("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
import dataSetGenerator as dsg

# keras imports
from keras.models import Sequential #used to initialize NN as sequence of layers
from keras.layers import Conv2D #used for convolutional step, and 2D as images are 2D (for videos, 3D would be used)
from keras.layers import MaxPooling2D #Used for pooling step
from keras.layers import Flatten #used to transform pooled maps into CNN input
from keras.layers import Dense #used to add fully connected in a classic ANN
from keras.layers import Dropout
from keras.optimizers import Adam #used as seen in Course 58 link of user apostolos
from keras.preprocessing.image import ImageDataGenerator #used for image preprocessing


# data

inputData = dsg.loadDataset("test_input.txt")
outputData = dsg.loadDataset("test_output.txt")
print("the size of the dataset is: ", inputData.shape, " of type: ", type(inputData))


# parameters

batchSize = 64
epochAmount = 100
imageSize=(inputData.shape[1], inputData.shape[2])


# CNN

'''cnn = Sequential()

cnn.add(Conv2D(32, (3, 3), input_shape = (16, 16, 1), activation = 'relu'))
# some weird problems with the input_shape

cnn.add(MaxPooling2D(pool_size = (2, 2)))

cnn.add(Flatten())'''


# ANN



'''cnn.add(Dense(units=64, activation='relu'))

cnn.add(Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])'''




# ANN standalone
cnn = Sequential()
cnn.add(Dense(units=64, activation='relu', input_dim=inputData.shape[1]*inputData.shape[2]))  # input
cnn.add(Dense(units=inputData.shape[1]*inputData.shape[2], activation='relu'))  # hidden
cnn.add(Dense(units=1, activation='sigmoid'))  # output

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

inputDataFlat = inputData.reshape(inputData.shape[0], inputData.shape[1]*inputData.shape[2])
outputDataFlat = outputData.reshape(outputData.shape[0], outputData.shape[1]*outputData.shape[2])


cnn.fit(inputDataFlat,
        outputDataFlat,
        batch_size=None,
        epochs=epochAmount,
        steps_per_epoch=(inputData.shape[0]//epochAmount),
        shuffle=True)

keepOpen=input()
