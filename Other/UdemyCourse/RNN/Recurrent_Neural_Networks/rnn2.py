# RNN
# PART 1: DATA PREPROCESSING









#______________________________________imports__________________________________

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1st change to other NN: we only train the network on the training set,
# not on both sets as with the ANN and CNN

try:
    dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
except:
    import os
    os.chdir(os.path.realpath("Other\\UdemyCourse\\RNN\\Recurrent_Neural_Networks"))
    dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")

training_set = dataset_train.iloc[:, 1:2].values
# [:, 1:2] means we take all the rows, and just column 1, but instead of saying
# [:, 1] we give a range, for some reason. .values makes it a numpy array

# print(type(dataset_train), type(training_set), training_set.size, training_set.shape)
# OUTPUT: <class 'pandas.core.frame.DataFrame'> <class 'numpy.ndarray'> 1258 (1258, 1)


#______________________________feature_scaling__________________________________

# two (best) ways to do feature scaling: stanardisation and nomralisation
# normalisation is recommended when using sigmoid function

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
# this makes so all our stock prices are between 0 and 1

# print(type(training_set_scaled), training_set_scaled.size, training_set_scaled.shape, training_set_scaled)
# <class 'numpy.ndarray'> 1258 (1258, 1) [[0.08581368] [0.09701243] ... --> correct results


#_____________________________special data structure____________________________

# we will no create a special structure with the stock prices:
# for each time t, the RNN will be able to look at the 60 stock prices before
# time t, and can look at the trends in these 60 timesteps.

# we will create x_train and y_train, x_train being the 60 stock prices before
# time t, and y_train wil be the estimated stock prices of the next day.

X_train = []  # will contain stock prices of index 0-59
y_train = []  # will contain stock price at index 60 which we want to predict

for i in range(60, 1258):  # 60 is used for the 60 days before, 1258 is total length of data
    X_train.append(training_set_scaled[i-60:i, 0])
    # we append to x the 60 previous prices, the ,0 is just to select the column
    y_train.append(training_set_scaled[i, 0])

# whats left is to transform our x and y python lists into numpy arrays

X_train, y_train = np.array(X_train), np.array(y_train)

# print(y_train, X_train)
# the output is a 2d matrix with each row being one day followed by the 60 previous ones


#_________________________reshaping the data____________________________________

# we add more dimension to our previous data by adding more predictors
# as of right now our only indicator is the "open" column

# we add another dimension to the numpy array by using the reshape function
# we simultaneously bring our data in the required input shape of keras
# first is batch size, second is timesetsp, last dimension can be other indicators, such as stock prices

# we do not add further indicators, thus the dimension is 1, but we could
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# print(X_train.shape) --> (1198, 60, 1)









# PART 2: BUILD RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

rnn = Sequential()

rnn.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# arguments explained
# units: how many units in the layer, we want high dimenionality --> lots of neurons
# return sequences: True, as our stacked LSTM layer returns sequences
# dimensions: only the last two, timesteps and indicators
rnn.add(Dropout(rate=0.2))

# we addd 3 more LSTM layer with dropout regularisation
rnn.add(LSTM(units=50, return_sequences=True))
rnn.add(Dropout(rate=0.2))

rnn.add(LSTM(units=50, return_sequences=True))
rnn.add(Dropout(rate=0.2))

rnn.add(LSTM(units=50))  # no return sequences, default values is false
rnn.add(Dropout(rate=0.2))

# final output layer
rnn.add(Dense(units=1))  # this is our stock price at time t+1

rnn.compile(optimizer="adam", loss="mean_squared_error")
# loss is no more binary crossentropy as with classification networks

rnn.fit(X_train, y_train, epochs=100, batch_size=32)

# Result:
# Epoch 1: loss: 0.0475
# Epoch 100: loss: 0.0016












# PART 3: PREDICTION AND VISUALISATION

#______________________________________imports__________________________________

# first we will get the real google stock price
# we import the test data / real data
# we do not use the try except statement anymore because if the above fails the
# directoy for the whole document has been changed so it cant possibily fail here
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# print(real_stock_price.shape)
# (20, 1) january 2017, 20 days without weekends

# Note we do not want/can expect the exact real shape in our result, but merely
# the trend, as the network is porbably not capable of detecting an irregulaity


# ______________________________________scaling___________________________________

# we will now use our rnn to predict the google stock prices from jan 2017

# we trained our model to predict stock price at time t based on 60 previous stock prices
# for each day, we need the 50 previous stock prices of the 60 days

# to get the 60 previous day we need the training set and the test set:
# training set for dates from december (training set, because these are known values)
# test set for dates from jnuary (test set because these are not used to train the network
# the network does not knwo these dates, but tries to preidct them, and thus only the
# test set contains the real values, not the training set)
#  --> we thus need to concatenatre both sets, to get the 60 previous inputs

# to do the concatenation, another problem arises:
# we can not concatenate the training_set with the real_stock_price because the
# training set is scalled, so we would need to scale the test set, which means
# that we need to modfy the actual test data which we should not do.
# the solution is to take the two original dataframes and scale them
# (we need to scale the inputs because the rnn was trained on the scaled inputs)

dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)
# ["Open"] because we want the column with the "open" values, axis=0 for vertical concatenation

# we now want to get the 60 previous inputs/days for each day
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
# len(dataset_total)-len(dataset_test) --> gives us january 3rd 2017, because it
# is the first day in both datasets, -60 to get the 60th previous day
# the upper bound is the last index of the whole dataset
# .values to make it a numpy array

# we need to reshape the data, to shape it like a numpy array
inputs = inputs.reshape(-1,1)

# whats left is to get the 3D format as previously, but before we also
# need to scale our inputs
inputs = sc.transform(inputs)  # we use the previously defined reshaping


X_test = []  # will contain stock prices of index 0-59

for i in range(60, 80):  # 60 is used for the 60 days before, 80 = 60 previous + 20 financial days of january
    X_test.append(inputs[i-60:i, 0])
    # we append to x the 60 previous prices, the ,0 is just to select the column

X_test = np.array(X_test)

# we now have again the structure where in each line of observations we have 60
# columns of the 60 previous days
# we just need to make this now a 3D data

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# now our strucutre is complete and we can start with the predictions

predicted_stock_price = rnn.predict(X_test)

# what is left to do is to inverse the scaling from our predicted resutls


predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# the variable now in the video contains data from 763-802 in a (20,1) array



# _________________________________visualising the results _____________________

plt.plot(real_stock_price, color = "red", label = "real google stock price")
plt.plot(predicted_stock_price, color = "blue", label = "predicted google stock price")
plt.title("google stock prices prediction")
plt.xlabel("time")
plt.ylabel("google stock price")
plt.legend()
plt.show()

# conclusion: the final graph is as expected in the video
