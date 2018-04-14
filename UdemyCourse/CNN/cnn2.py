import os
os.chdir("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\UdemyCourse\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Convolutional_Neural_Networks")
#the above code is used to change to the right directory


#                                                                                                           
#                                             Course-44                                                     
#                                                                                                           

#CNN basically just ANN with the additioanl Convolutional method and layers --> used to preserve structure and spatial features
#goal of network is to classify images to differentiate between dogs and cats



#                                                                                                           
#                                             Course-45                                                     
#                                                                                                           

#we could use the standard way to import the images, but we used a different approach instead and choose Keras to import them
#in order to use Keras, we only need to our dataset to have a certain structure:
# 1) separate images in test and training folder (same 80/20 relation as before
# 2) make cat and dog folder inside each of these folders which are filled with images with descriptive names ("dog-123.jpg")

#for this CNN, contrary to our ANN, we dont need DATA PREPROCESSING OF CATEGORICAL DATA, SPLITTING INTO DIFFERENT SETS



#                                                                                                           
#                                             Course-46                                                     
#                                                                                                           

from keras.models import Sequential #used to initialize NN as sequence of layers
from keras.layers import Conv2D #used for convolutional step, and 2D as images are 2D (for videos, 3D would be used)
from keras.layers import MaxPooling2D #Used for pooling step
from keras.layers import Flatten #used to transform pooled maps into CNN input
from keras.layers import Dense #used to add fully connected in a classic ANN



#                                                                                                           
#                                             Course-47                                                     
#                                                                                                           

#same as before, the classifier is created to afterwards classify the images

classifier=Sequential() #this initializes our CNN



#                                                                                                           
#                                             Course-48                                                     
#                                                                                                           

#recap of CNN: Input --> Convolution --> Max Pooling --> Flattening --> ANN/Full connection --> Output

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#for our ANN, we used Dense function, but for the CNN we used convolution2D
#convolution2d arguments:

#nb_filter = number of feature detectors --> number of feature maps created
#nb_row, nb_column= dimension of the feature detector table (which goes over image)

#nbfilter: best method is to start with 32 and then gradually increase the power of 2s

#input shape = important, shape of input image
#--> as our images have different formats, we need resize them all to the same format --> done in the image preprocessing part
#our images are transfomed into RGB 3D arrays --> 3 for 3D, 256x256 are dimension

#important: order of input shapes is different in theano backend (docstring/documentation) and tensorflow backed (the one we use)

#activation function: here used  to prevent negative pixel values in our feature maps -> to increase non-linearity



#                                                                                                           
#                                             Course-49                                                     
#                                                                                                           

#this step is about the pooling step --> reducing the size of the feature map --> the result is the pooling layer
#we use max pooling to reduce input size of ANN to improve effiency and speed without increasing performance

classifier.add(MaxPooling2D(pool_size = (2, 2))) #we add the pooling layer, and the parameters are the size of the square going over the feature map

classifier.add(Conv2D(32, (3, 3), activation = 'relu')) #lesson54
classifier.add(MaxPooling2D(pool_size = (2, 2))) #lesson54

classifier.add(Conv2D(64, (3, 3), activation = 'relu')) #lesson54
classifier.add(MaxPooling2D(pool_size = (2, 2))) #lesson54



#                                                                                                           
#                                             Course-50                                                     
#                                                                                                           

#now comes the third step: flattening: putting the pooled feature maps in the input vector
#structure of image is not lost due to feature maps and pooling maps --> high numbers are always preserved in every step
#without the pooling and feature mapping, we would loose the relation between a pixel and its surrounding pixels

classifier.add(Flatten()) #no need for parameters, done automtically py keras

#now we just need to create the classic ANN



#                                                                                                           
#                                             Course-51                                                     
#                                                                                                           

#in this last step, we create the ANN to work the data in the vector created in the flatten step

classifier.add(Dense(units=128, activation='relu')) #hidden layer
                     
#output_dim = units = number of nodes in hidden layer --> not too small number and not too big, number around 100 is good choice, 2^x often recommended

classifier.add(Dense(units=1, activation='sigmoid')) #output layer

#sigmoid function and 1 used due to binary outcome



#                                                                                                           
#                                             Course-52                                                     
#                                                                                                           

#now we compile the NN by choosing a stochastic gradient descent, loss function and a performance metric

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#optimizer = stochasitc gradient algorith, we choose same as before
#loss = loss functino, we use binary cross entropy, as its used in classification models and for binary outcomes



#                                                                                                           
#                                             Course-53                                                     
#                                                                                                           

#this step is about image preprocessing where we fir our CNN to our image dataset
#we use built in keras methods to do it instead of doing it manually
#we also want to prevent overfitting (great results in training set and poor performace on test set)

#we go to https://keras.io/preprocessing/image/

#how will image preprocessing prevent overfitting?
#a lof of images are required --> 10.000 is not enough to prevent overfitting
#we those use keras tools to create batches, and in each, random transformations are applied to the images (rotating, flipping, ...)
#result is more diverse images and more data --> called image augmentation, as training set is increased
#summary: preprocessing allows to increase dataset without adding more images

#we use the flow from directory and copy the code directly into our (this) file

from keras.preprocessing.image import ImageDataGenerator #used to allow these modifications

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True) #creation of image generation class

#rescale rescales image, shearing is when pixels are moved in fixed direction in a parallel way
#zoom is a random kind of zoom and flip flips uses the image
#we could use more available methods, but this should be enough
#we keep the default values for the parameters that keras suggest

test_datagen = ImageDataGenerator(rescale = 1./255) 

#we dont need to change anything here

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

#this step creates the augmented training set from our preprocessing modifications on the images
#we need to define the direcoty we extract the data from (relative path is enough if we run python in the right working directoy)
#the target size must be the value expect and previously defined by our CNN
#batch size is batch size, 32 is ok
#class mode is binary, as we expect a binary result

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#we need to remember to change the path  to use the test set, and the rest stays the same

#same, but for test set

classifier.fit_generator(training_set,
                         steps_per_epoch = (8000/32),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = (2000/32),
                         workers=8)

#executing this start the NN and takes some time

#fit trains the CNN on the training set and test its performance on the test set
#samples per epoch = number of images in our training set
#nb_epochs is the amount of epochs
#validation data is the test set we want to use to check the performance, and validation steps is the number of images in this set


#update; as seen in https://www.udemy.com/deeplearning/learn/v4/questions/3672714, the steps have to be divided by batch size in netwer keras
#also workers and multiprocessing have been added (not sure, as im using GPU)



#                                                                                                           
#                                             Course-54                                                     
#                                                                                                           

#previous model reached training accuracy of 81% and test accuracy of 76%, not ideal, and we try to improve it now
#how? --> make it deeper --> 2 options: adding convoplutional layer to CNN / adding hidden layer to ANN in CNN --> we do first

#to do it, we just copy and paste the Conv2D line behind the pool line and then also the pool line behind that

#we also need to change the input shape to the new dimension of the pooled feature map
#--> to do so, we just remove the input shape parameter and let keras use the deafult one

#by adding a third vonv layer with 64 value, we can increase accuracy even more (not done in video, but I did it as suggested)

#as a result, the accuracy has improved to: training: 90%, test: 83%

#last change: we upped the resolution of the images from 64 to 256 to increase accuracy even more
#this step was not done in the video and only suggested to do with a GPU

#needs to be changed in initiation of NN and in the image preprocessing (now its 256 everywhere)

#final accuracy: training 91, test 82%

#because of this poor improvement, image size has been reduced to 64 to allow for quicker tests



#                                                                                                           
#                                             Course-56/57                                                  
#                                                                                                           

#now as usual we just add the functionaliy to input our sample and let the NN predict it
#we can use any image or the provided one in the new prediction subfolder of the dataset

#to preprocess the image we use numpy and the preprocessing tool from keras
import numpy as np
from keras.preprocessing import image

test_image=image.load_img('dataset/single_prediction/mila1.jpg', target_size = (64, 64))

#now we need to add another dimension to the image: we need to split it into 3 dimensional color arrays

test_image=image.img_to_array(test_image)

#now our image is in the form of an array and we are ready to use the predict function
#but before, we need to add a further dimension which contains the batch (in this case, just

test_image=np.expand_dims(test_image, axis=0) #add dimension to array at the index (axis variable)

#thus we are now ready to predict:

result=classifier.predict(test_image)

#now printing the result gives us a value, 1 or 0, but we need to find out to what it corresponds

training_set.class_indices

#executing this tells us wheter 1 or 0 is dog or cat

if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'

#the same thing but more nice
    



#                                                                                                           
#                                             Course-58                                                  
#                                                                                                           

#we are improving the CNN, but not in this file but in the cnn3.py
#only change that we implemnted here: added workers=8 argument to fit_generator to speed up process

#                                                                                                           
#                                             additional info                                               
#                                                                                                           

#to run files, best is to launch shell from specified directory and then write python and then import cnn.... and it starts
