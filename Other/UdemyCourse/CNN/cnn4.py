#This version has no/less annotation and improved/experimental performance

import os
os.chdir("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\UdemyCourse\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Convolutional_Neural_Networks")
#above is for the stanard training set, below is for the custom matrix set
#os.chdir("D:\\School\\BICS BSP 2")

from keras.models import Sequential #used to initialize NN as sequence of layers
from keras.layers import Conv2D #used for convolutional step, and 2D as images are 2D (for videos, 3D would be used)
from keras.layers import MaxPooling2D #Used for pooling step
from keras.layers import Flatten #used to transform pooled maps into CNN input
from keras.layers import Dense #used to add fully connected in a classic ANN
from keras.layers import Dropout
from keras.optimizers import Adam #used as seen in Course 58 link of user apostolos
from keras.preprocessing.image import ImageDataGenerator #used for image preprocessing


#CNN                                                                                                                                                                

imageSize=32

classifier=Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

#ANN                                                                                                                                                                

classifier.add(Dropout(0.6))

classifier.add(Dense(units=128, activation='relu')) 

classifier.add(Dropout(0.5))

classifier.add(Dense(units=128, activation='relu')) 

classifier.add(Dropout(0.3))

classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])

#image preprocessing                                                                                                                                                


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True) 

test_datagen = ImageDataGenerator(rescale = 1./255) 

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (imageSize, imageSize),
                                                 batch_size = 64,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (imageSize, imageSize),
                                            batch_size = 64,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = (8000//64),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = (2000//64),
                         workers=12)

#this starts the NN. prediction code also removed

#LOG of improvements                                                                                                                                                    

#no change: train_acc=92, test_acc=83, time=39s/epoch

#adding
""" workers=12,
    max_q_size=100,""" #to the fit_generator: train_acc=90, test_acc=83, time=8s/epoch

#changing image size to 128, adding dropout and more hidden layers in the CNN and removing max_q_size
#most impressive consquence: highest GPU load ever (was 8 in CNN2, 15 before this and now around 30%)
#train_acc=87, test_acc=83, time=13s/epoch

#slightly chnaning the order of the ANN layers, and setting workers to 16
#----------> - 8s 31ms/step - loss: 0.2349 - acc: 0.9031 - val_loss: 0.4581 - val_acc: 0.8285

#changing ANN again, other dropout value, up the epoch number to 35: high GPU usage, but strong overfitting
#---------> - 13s 53ms/step - loss: 0.1343 - acc: 0.9457 - val_loss: 0.5510 - val_acc: 0.8410


#changing optimizer from "adam" to Adam(lr=1e-3) (as seen in apostolos post in the homework solution to question 53)
#also increasing dropout due to increased overfitting in previous model, and decreasing epochs to 25 (too slow)
#increasing workers to 32 --> this last change did not do much, GPU usage still at 30-40%
#---------> - 13s 52ms/step - loss: 0.2847 - acc: 0.8819 - val_loss: 0.4078 - val_acc: 0.8470

#increasing batch size from 32 to 128
#--------> - 12s 190ms/step - loss: 0.2766 - acc: 0.8823 - val_loss: 0.3899 - val_acc: 0.8400






