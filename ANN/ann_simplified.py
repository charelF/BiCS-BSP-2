import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split  # split ds
from sklearn.preprocessing import StandardScaler  # feature scaling
import keras
from keras.models import Sequential  # NN creation
from keras.layers import Dense, Dropout  # NN creation
from sklearn.metrics import confusion_matrix  # output evalutation
from keras.wrappers.scikit_learn import KerasClassifier  # k-fcv
from sklearn.model_selection import cross_val_score  # k-fcv

# import dataset
dataset = pd.read_csv('test.csv')
#dataset = pd.read_csv('Churn_Modelling.csv')

a = 13
""" a is the length of the input matrix.
    There is a problem: normally the NN should only put out a good accuracy (80%)
    When a is 1024, but in our case it does it for all kinds of values that a
    can have, which is strange and unwanted

    POSSIBLE CAUSE we left out the one-hot encoding and other data transformations
    which should technically not be neede but maybe they are and that causes the error
"""
b= 0
#b is only needed for the churn modeling, where we set b to 3
#b=3

# sorting the data
X = dataset.iloc[:, b:a].values  # x = input matrix
y = dataset.iloc[:, a].values  # y = expected output vector

# encoding the dataset
# this part is intentionally left out as it is used to encode strings and non
# binary values to binary values, a problem which we dont have with our dsself.
# same goes for the one-hot-encoding, which should technically not be neeeded

# splitting the dataset into test and training set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Neural Network implementation with keras
classifier = Sequential()

classifier.add(Dense(512,
                     kernel_initializer='uniform',
                     activation='relu',
                     input_dim=a))  # input layer + first hidden layer
classifier.add(Dropout(p=0.1))

classifier.add(Dense(128,
                     kernel_initializer='uniform',
                     activation='relu'))  # 2nd hidden layer
classifier.add(Dropout(p=0.1))

classifier.add(Dense(1,
                     kernel_initializer='uniform',
                     activation='sigmoid'))  # output layer

classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])  # compilation of NN

# Starting the traiining of the above defined network
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Evaluation of the NN
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)  # True or False

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# OPTIONAL: Making a prediction with the NN
"""
new_prediction=classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40,
                                                          3, 60000, 2, 1, 1,
                                                          50000]])))
new_prediction = (new_prediction > 0.5)
"""

# K-fold Cross Validation implementation


def build_classifier():
    classifier.add(Dense(512,
                         kernel_initializer='uniform',
                         activation='relu',
                         input_dim=a))  # input layer + first hidden layer
    classifier.add(Dropout(p=0.1))

    classifier.add(Dense(128,
                         kernel_initializer='uniform',
                         activation='relu'))  # 2nd hidden layer
    classifier.add(Dropout(p=0.1))

    classifier.add(Dense(1,
                         kernel_initializer='uniform',
                         activation='sigmoid'))  # output layer

    classifier.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])  # compilation of NN
    return classifier


classifier = KerasClassifier(build_fn=build_classifier,
                             batch_size=10,
                             epochs=100)

accuracies = cross_val_score(estimator=classifier,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             n_jobs=1)

mean = accuracies.mean()
variance = accuracies.std()
