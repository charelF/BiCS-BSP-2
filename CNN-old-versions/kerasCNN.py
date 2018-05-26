#This version has no/less annotation and improved/experimental performance

import os
#os.chdir("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\UdemyCourse\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Convolutional_Neural_Networks")
#above is for the stanard training set, below is for the custom matrix set
#os.chdir("D:\\School\\BICS BSP 2")
os.chdir("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\UdemyCourse\\CNN")

from keras.models import Sequential #used to initialize NN as sequence of layers
from keras.layers import Conv2D #used for convolutional step, and 2D as images are 2D (for videos, 3D would be used)
from keras.layers import MaxPooling2D #Used for pooling step
from keras.layers import Flatten #used to transform pooled maps into CNN input
from keras.layers import Dense #used to add fully connected in a classic ANN
from keras.layers import Dropout
from keras.optimizers import Adam #used as seen in Course 58 link of user apostolos
from keras.preprocessing.image import ImageDataGenerator #used for image preprocessing

#parameters                                                                                                                                                         

imageSize=32

batchSize=64

epochAmount=50


#CNN                                                                                                                                                                

classifier=Sequential() 

classifier.add(Conv2D(32, (3, 3), input_shape = (imageSize, imageSize, 3), activation = 'relu')) #convolutional layer

classifier.add(MaxPooling2D(pool_size = (2, 2))) #pooling layer

classifier.add(Flatten())


#ANN                                                                                                                                                                

classifier.add(Dense(units=64, activation='relu')) #hidden layer

classifier.add(Dense(units=1, activation='sigmoid')) #output layer

classifier.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy']) #training method


#image preprocessing                                                                                                                                                

"""
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True) 
""" #we do not add filters to our images

train_datagen = ImageDataGenerator(rescale = 1./255) #note: if our
# dataset is only 0 and 1s, we dont need the rescale !!!!

test_datagen = ImageDataGenerator(rescale = 1./255) 

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (imageSize, imageSize),
                                                 batch_size = batchSize,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (imageSize, imageSize),
                                            batch_size = batchSize,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = (8000//batchSize),
                         epochs = epochAmount,
                         validation_data = test_set,
                         validation_steps = (2000//batchSize),
                         workers=12)






