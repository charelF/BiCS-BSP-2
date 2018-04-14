#running this file necessites python 3.6.4, numpy, matplotlib, pandas, scipy, scikit-learn/sklearn, tensorflow, keras, theano


#                                                                                                                           
#                                               COURSE-18                                                                   
#                                                                                                                           


# Importing the libraries
import numpy as np
import matplotlib as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv') #print(variable) is the same as the variable explorer form spyder 3
X = dataset.iloc[:, 3:13].values #form our data of the csv table in the dataset, the values from columns 3 to 12 (13, as last one is not included)
#is the data which indicates us wheter a user wants to leave the bank or not
# this line creates our matrix of features
#print(X) shows us our independent variables in the form of a matrix
y = dataset.iloc[:, 13].values #creates our independent variable vector --> the results, the binary outcomes



# Encoding categorical data
#if i understand it correctly, this code is used to encode the data, for example as there are strings in our dataset which need be transformed
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder() # ..._1 is for the countries
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #converts the strings "france, spain, germany, ..." into numbers --> print(X) shows the changes

labelencoder_X_2 = LabelEncoder() # ..._2 is for the gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #btw the 2 in the parentheses is the index of the data we want to transfomr

onehotencoder = OneHotEncoder(categorical_features = [1]) #this code defines that the value 0,1,2 have no particular importance and are only
#important to differentiate the countries. In other words, this tells us that the country 2 is not in any way better than the country 1, it is
#just different
#the way this is achieved is by splitting the colomn of countries with possible values 0,1,2 into three columns of possible value 1 or 0.
#print(X) shows us that
X = onehotencoder.fit_transform(X).toarray()
#at this point, all our data is in the form of numbers.

X=X[:, 1:] #we remove one of the dummy variables created in the previous step



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) #the newly created varaibles x_train, X_test etc contain the scaled values of X. --> data is now preprocessed
X_test = sc.transform(X_test) 



#                                                                                                                           
#                                               COURSE-19                                                                   
#                                                                                                                           



import keras #keras will build the deeep neural network based on tensorflow / alternatively the theano backend could be used
from keras.models import Sequential #used to initialize NN
from keras.layers import Dense #used to create layers in NN
from keras.layers import Dropout #-->Course-30



#                                                                                                                           
#                                               COURSE-20                                                                   
#                                                                                                                           



#first step of creating the NN: initilasing by defining it as a sequence of layers
#we create a neural network that has the role of a classifier, as we want a binary result

classifier = Sequential() #this initialises our ANN



#                                                                                                                           
#                                               COURSE-21                                                                   
#                                                                                                                           


#adding the input layer and the first hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))

classifier.add(Dropout(p=0.1))#-->Course-30

#the arguments of the dense function are the parameters: how the weights are updated, whcih actiovation function did we use, number of nodes
#of first hidden layer and input layer etc.. --> to get a better overview, typing Dense( --> shows use the docstring !!! works even in comment
    
#output dim= number of nodes of layer we are adding in add function = hidden layer --> how many should we add? difficult to answer
#but tip is to add the average of the nodes of the input layer and the output layer --> this is 11+1/2=6 in our case.
#!!!!!!!! also in the newest version of keras, this parameter is called (units)(not sure, but since it is the first
#parameter, not giving it a name seems to work too

#init is the function to initialise the weights
#!!!!!!!!!!!!!!!!!!!!!!!!!!   apparently called kernel_initializer with the newest version of keras (version 2.0)

#activation is the activation formula. We are choosing the rectifier function, which is called relu

#input_dim is another obligatory argument to add and specifies the number of input nodes of the input layer



#                                                                                                                           
#                                               COURSE-22                                                                   
#                                                                                                                           

#in this step we are adding more hidden layers as we are trying to create a DEEP neural networks thus using multiple hidden layers

classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))

classifier.add(Dropout(p=0.1))#-->Course-30

#input dim can now be left out as we know the number of nodes from the previous layer



#                                                                                                                           
#                                               COURSE-23                                                                   
#                                                                                                                           

#we now added 3 layers, so now we are adding the output layer

classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

#the output layer is only one node, and we will also change the activation function to the sigmoid function to get a probability for the outcome
#if there were multiple output layers, we should use softmax as activation function




#                                                                                                                           
#                                               COURSE-24                                                                   
#                                                                                                                           

#now we need to compile the NN --> applying stochastic gradient descent on the whole NN

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#contains several arguments:
#optimizer: which algorithm to use to find the perfect weights. We use the stochastic radient descent, and more precisely the adam verison
#loss function used to optimise the parameters, we are using the one for a binary outcome
#metrics: a criterion used to evaluate the model --> used after each bach to improve accuracy of the model



#                                                                                                                           
#                                               COURSE-25                                                                   
#                                                                                                                           

#now we fit the ANN through the training set

classifier.fit(X_train, y_train, batch_size=10, epochs=100)
               
#first arguments =the dataset used to train the classifier = trainig set
#x_train = matrix of features of the observations of the trainig set, y_train = actual oputcomes for all the observations
#batch_size, epochs: no rule to chosse which is the best, we choose 10 and 100

#!!!!!!!!!! nb_epoch has been renamed to epochs due to a warnig/newer version

#this starts the NN



#                                                                                                                           
#                                               COURSE-26                                                                   
#                                                                                                                           

#we are now creating a variable with the predicted probabilities

y_pred = classifier.predict(X_test)

#then we are changing the variable to show a bool True or False based if the probaility is above 50%

y_pred = (y_pred > 0.5)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)

#this is a matrix which contains the correct predictions (rows-colomns: 0-0, 1-1) and the wrong predictions (0-1, 1-0)



#                                                                                                                           
#                                               COURSE-27,28                                                                
#                                                                                                                           

#Using the created NN to predict whether a customer will leave the bank or not

new_prediction=classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
#we need to put the new information into and array, more precisely into a horizontal vector --> we are doing this by using double brackets
#when we start to input the data, we must watch out that they are in the same order as in the amtrix of features
#we also need to watch out to input the data in the same form as before, ie using dummy variables etc
#--> to do this, open/print the dataset and matrix of features(X) variables to check in which form the values are
#--> for example, check another customer who lives in France, and check his values for the columns

#we also cant forget to scale the values afterwards --> we take the sc object, as it was fitted to our training set

new_prediction = (new_prediction > 0.5) #finally we do this to get a yes no answer whether the customer will leave the bank

#inputting the above two lines would result in a warning as the inputed values need to be floats. In any way, 
#transforming the first values of the prediction into a float will do the trick

#finally, printing(new_predition) gives us the answer us to whwther the new customer will leave the bank or not



#                                                                                                                           
#                                               COURSE-29                                                                   
#                                                                                                                           

#we want to evaluate the performace of our NN --> how accurate is it really?
#we want to improve the bias-variance tradeoff: find model that is simultaneously accurate and has a low variance over multiple trainings
#so far, we split our training data between training set and test set and trained the model on the training set and tested it on the test set
#--> correct way, but not best, as we get variance problem: we might get very different accuracies depending on the test set with the same model
#--> jusdging performace of test set on one accuracy on one test set = not most relevant --> kfcv improves this a lot

#we use a technique called k-Fold Cross Validation

#before starting with this part, we need to run the code up to Course 19 not included

#we need to combine scikit-learn and keras, as our NN is on keras, but the k-fold cross validation uses sci-kit

from keras.wrappers.scikit_learn import KerasClassifier #this imports the neeeded functionality to combine both modules

from sklearn.model_selection import cross_val_score #this imports the k fold cross validation function

from keras.models import Sequential
from keras.layers import Dense #these two also need be imported if previous code not executed

def build_classifier():
    #this function builds the ANN classifier exactly as we build it in the above sections (copy paste from above)
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

#the classifier inside the function is a local variable
#we thus create a global variable classifier, which will be build wih k-fcv on 10 different training folds and measuring model performance on one test fold
    
classifier=KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)

accuracies=cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=1) 
#this new variable contains the 10 accuracies returned by the k-fcv with k=10 (sef explanatory arguments not explained)
#cv= number of folds
#n_jobs=number of CPUs to use for the computation (important, as this step takes 10times longer than the training before (10 accuracies), -1 means all CPUs



#!!!!!!!!!!!!! APPARENTLY WINDOWS DOES NOT ALLOW FOR ALL CORES TO BE USED SO WE HAVE TO SET N_JOBS TO 1 AND ONLY USE 1 CORE, THUS TAKING LONGER !!!!!!!!!!!!!

#when the exceutino is over, we can check out the vector of accuracies with the variable accuracies

mean=accuracies.mean() #this gives us the mean of accuracy of 10 epochs, which is more precise and useful than the accuracy after only 1 epoch
variance=accuracies.std() #gives variance (see beginning of Course 29 for explicit explanation of variance etc)



#                                                                                                                           
#                                               COURSE-30                                                                   
#                                                                                                                           

#dropout regularization = solution to over-fitting = when model is trained to much on training set resulting in lower performance in test set
#can be observed when there is a large difference of accuracy between training and test set/ or ehen the variance is high

#how dropout works: in each iteration of the training, some neurons are randomly disabled to prevent them from becoming to dependant on each other
#thus the neurons work more independently and are less dependant of other neurons --> prevents them from learning too much --> prevents overfitting
#the changes made in this section are in the existing code and referenced with a comment behind their name

#amount of overfitting = amount of layers we should apply the dropout to

#dropout applied to first hidden layer: arguments:
#p: between 0 and 1, fraction of the neurons we want to drop/disable each iteration (start with low(0.1), dont go over 0.5)



#                                                                                                                           
#                                               COURSE-31                                                                   
#                                                                                                                           

#we are tuning the NN by using parameter tuning
#which parameters?: 1) learned ones: weights, 2) fixed ones: hyperparameters: number of epochs, batchsize, optimiser, number of neurons, ...
#maybe changing these fixed ones will improve the accuracy of the NN

#gridserach = technique we use to find the best values for the hyperparameters
#--> implementing it is similar to implementing kfcv

#we copy most of the code from Course 29 to start off


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV #we change the class/function from kfcv to gridsearch
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    #we use optimizer as argumnet for function to change it later, explained below (row 331-332)
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier)
#here we remove the batch and epoch arguments as these are the exact kind of arguments we want to tune with gridsearch

#now we are ready to implement the code used for gridsearch

parameters={'batch_size': [20, 25],
            'epochs': [300, 500, 700],
            'optimizer': ['adam', 'rmsprop']}
#dictionariy with key value pair being the hyperparameters we want to tune and the possible values we want to try to use for them.
#gridsearch will then use all this different combinations, train them and output the best combination
                
#our third pair is the optimizer. To acces it, we give the optimizer as an argument to the build_classifier function above, which we then can change more easily
#we mustnt forget to replace the 'adam' in the function body by the variable optimizer so we can change it

#now we can start with gridsearch

grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10) #cv=number of folds

#our gridsearch object is nearly finish, now we must fit it to the training set

grid_search=grid_search.fit(X_train, y_train)

#now our gridsearch is implemented and ready to use. We will add some code that tells us the best accuracy and parameters

best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_


#we are now ready to execute:
#!!! before execution, at least run everything up to Course 19 not included, or alternatively, just run everything

#WARNING: THIS WILL TAKE EXTREMELY LONG, UP TO SEVERAL HOURS.



#when it is finished, the parameters best parameter and best accuracy give us the long awaited values.


#THIS IS THE END OF THE ANN SECTION
#SOME OF THE VALUES OF THE LAST COURSE MIGHT BE CHANGED DUE TO THE HOMEWORK (COURSE 32)
                
#result with our data: print(best_accuracy, best_parameters)
#0.852375 {'batch_size': 25, 'epochs': 300, 'optimizer': 'rmsprop'}


























