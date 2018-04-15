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

# CNN

cnn = Sequential()

cnn.add(Conv2D(32, (3, 3), input_shape = inputData.shape, activation = 'relu'))

cnn.add(MaxPooling2D(pool_size = (2, 2)))

cnn.add(Flatten())

cnn.add(Dense(units=64, activation='relu'))

cnn.add(Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])

cnn.summary()

cnn.fit(inputData,
        outputData,
        epochs=100,
        validation_split=0.2)

keepOpen=input()
