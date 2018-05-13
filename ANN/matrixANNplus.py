# personal module imports

import sys
sys.path.append("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
import dataSetGenerator as dsg
import glob


# keras imports

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# data


inputData = dsg.loadDataset(glob.glob("*_input.txt")[0])
outputData = dsg.loadDataset(glob.glob("*_output.txt")[0])

inputDataFlat = inputData.reshape(inputData.shape[0], inputData.shape[1]*inputData.shape[2])
outputDataFlat = outputData.reshape(outputData.shape[0], outputData.shape[1]*outputData.shape[2])


# # ANN
#
# ann = Sequential()
#
# ann.add(Dense(units=100, activation='relu', input_dim=inputData.shape[1]*inputData.shape[2]))  # input
#
# ann.add(Dropout(rate=0.4))
#
# ann.add(Dense(units=10, activation='relu'))  # hidden
#
# ann.add(Dropout(rate=0.4))
#
# ann.add(Dense(units=1, activation='sigmoid'))  # output
#
# ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# ann.summary()
#
# ann.fit(inputDataFlat,
#         outputDataFlat,
#         epochs=25,
#         validation_split=0.2)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def buildAnn(optimizer):
    ann = Sequential()
    ann.add(Dense(units=100, activation='relu', input_dim=inputData.shape[1]*inputData.shape[2]))
    ann.add(Dropout(rate=0.4))
    ann.add(Dense(units=10, activation='relu'))
    ann.add(Dropout(rate=0.4))
    ann.add(Dense(units=1, activation='sigmoid'))
    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return ann

ann = KerasClassifier(build_fn=buildAnn)

parameters = {'epochs': [30, 100], 'optimizer': ['adam', 'rmsprop']}

gridSearch=GridSearchCV(estimator=ann,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=3)  # cv = number of folds

gridSearch = gridSearch.fit(inputDataFlat, outputDataFlat, validation_split=0.2)

bestParameters = gridSearch.best_params_  # gives us best parameters
bestAccuracy = gridSearch.best_score_  # gives us the highest accuracy

# for explanations, see Udemy ann2.py

""" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
vialicht dat do nach anbauen:
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
"""
















keepOpen=input()

# Summary: This network is completely working and can be tuned, modified an
# studied to understand its behvaiour.
