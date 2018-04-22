# personal module imports
import sys
sys.path.append("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
import dataSetGenerator as dsg
import glob  # for the dataset name detection

# keras imports
from keras.models import Sequential  # used to initialize NN as sequence of layers
from keras.layers import Conv2D  # used for convolutional step, and 2D as images are 2D (for videos, 3D would be used)
from keras.layers import MaxPooling2D #Used for pooling step
from keras.layers import Flatten #used to transform pooled maps into CNN input
from keras.layers import Dense #used to add fully connected in a classic ANN
from keras.layers import Reshape
from keras.layers import Dropout
from keras.optimizers import Adam #used as seen in Course 58 link of user apostolos
from keras.preprocessing.image import ImageDataGenerator #used for image preprocessing

from keras.wrappers.scikit_learn import KerasClassifier  # used to wrap/combine scikit-learn and Keras
from sklearn.model_selection import GridSearchCV  # used for the hyperparameters

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

def buildCnn(D, batchsize):
    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), input_shape = (inputDataShape[1], inputDataShape[2], 1), activation = "relu"))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))

    if D:
        cnn.add(Conv2D(32, (3, 3), activation = "relu"))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

    cnn.add(Flatten())
    cnn.add(Dense(units=64, activation="relu"))
    cnn.add(Dense(units=1, activation="sigmoid"))
    cnn.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    cnn.summary()
    return cnn

# Hyperparameter optimization

cnn = KerasClassifier(build_fn=buildCnn)

parameters = {"D":[True], "epochs":[70], "batchsize":[10]}
# this article gives a list of parameters we can use:
# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

gridSearch=GridSearchCV(estimator=cnn,
                         param_grid=parameters,
                         scoring="accuracy",
                         cv=3)

gridSearch = gridSearch.fit(inputData, outputData, validation_split=0.2)

bestParameters = gridSearch.best_params_  # gives us best parameters
bestAccuracy = gridSearch.best_score_  # gives us the highest accuracy

print("\n\n\n========================= Conclusion: =========================\n")
print("best parameters were: ", gridSearch.best_params_)
print("best accuracy was: ", gridSearch.best_score_)
print("details: \n\n", gridSearch.cv_results_)

# for explanations, see Udemy ann2.py

keepOpen=input()
