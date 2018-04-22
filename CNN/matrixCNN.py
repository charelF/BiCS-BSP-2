# personal module imports
import sys
sys.path.append("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
import dataSetGenerator as dsg
import glob  # for the dataset name detection

# keras imports
from keras.models import Sequential #used to initialize NN as sequence of layers
from keras.layers import Conv2D #used for convolutional step, and 2D as images are 2D (for videos, 3D would be used)
from keras.layers import MaxPooling2D #Used for pooling step
from keras.layers import Flatten #used to transform pooled maps into CNN input
from keras.layers import Dense #used to add fully connected in a classic ANN
from keras.layers import Reshape
from keras.layers import Dropout
from keras.optimizers import Adam #used as seen in Course 58 link of user apostolos
from keras.preprocessing.image import ImageDataGenerator #used for image preprocessing


# data

inputData = dsg.loadDataset(glob.glob("*_input.txt")[0])
outputData = dsg.loadDataset(glob.glob("*_output.txt")[0])

print("\n\n\n========================= DataSetInfo: =========================\n")
print("Size:                ", inputData.shape)
print("Type:                ", type(inputData))
print("InputName:           ", glob.glob("*_input.txt")[0])
print("InputDescription:    ", dsg.loadDatasetDescription(glob.glob("*_input.txt")[0]))
print("OutputName:          ", glob.glob("*_output.txt")[0])
print("OutputDescription:   ", dsg.loadDatasetDescription(glob.glob("*_output.txt")[0]))
print("\n================================================================\n\n\n")

inputDataShape = inputData.shape
# inputDataShape = tuple(amount of matrices, columns, rows)
inputData = inputData.reshape(*inputDataShape, 1)
outputData = outputData.reshape(inputDataShape[0], 1)


# CNN


cnn = Sequential()

cnn.add(Conv2D(32, (3, 3), input_shape = (inputDataShape[1], inputDataShape[2], 1), activation = 'relu'))

cnn.add(MaxPooling2D(pool_size = (2, 2)))

# cnn.add(Conv2D(20, (3, 3), activation="relu"))
# cnn.add(MaxPooling2D(pool_size = (2, 2)))
# --> addding one more conv+pool layer drastically increases accuracy.

cnn.add(Flatten())

cnn.add(Dense(units=64, activation='relu'))

cnn.add(Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])

cnn.summary()

cnn.fit(inputData,
        outputData,
        epochs=20,
        validation_split=0.2)

keepOpen=input()
