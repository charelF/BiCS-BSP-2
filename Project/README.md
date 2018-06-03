# This is the project folder of the BiCS-BSP-2

## datasets folder
This folder contains a few datasets and their visual representations and was created to not convolute the project directory with the datasets. Newly created datasets will still be in the project folder, and datasets need also to be in the project folder for the networks to be able to train on them. (The reason why it is not possible to use them in a subdirectory is because this would be difficult to implement when considering that all the code should work on Windows
and UNIX OS's

## CNN-simple, ANN-simple
These are our two main neural networks, an artificial and a convolutional neural networks. To train them, it is necessary that corresponding input and output dataset text files are in the same folder as the neural networks

## DatasetGenerator
Contains the create and loadDataset functions

## MatrixGenerator
Contains the Matrix, EmptyMatrix and FeatureMatrix classes

## DatasetCreationTool
A simple script to create datasets very quickly by executing the python file. It uses the MatrixGenerator

## DatasetVisualizer
A simple script to create a visual representation of any dataset currently in the project folder

## test_input.txt, test_output.txt, test_input.png
These are sample datasets, which are not meant to be used to train an neural network.

## ANN-HPO, CNN-HPO:
these are the Hyperparameter optimised versions of the Neural Networks. They are only briefly mentioned in the report as we did not manage to complete them to the level
we desired. The CNN-HPO for example only trys out different network strucutures (i.e. which types of layers and hown many of them). But due to the exponential nature of
the code, we have 1024 different networks. (Using Gridsearch and setting the cross validation paramter to 3 (as is per default), this number would rise to over 3000
runs) This means that if the CNN-simple network takes 1 time unit for a given dataset, the CNN-HPO dataset takes over 3000 time units. Altough this is still a feasible
time, it get's worse when we consider that the CNN only optimises a low amount of values. 

To demonstrate the complexity, we included much more parameters to be optimized in the ANN network. In its current state, the ANN would technically need 2^68 time units
As this is not physically possible to do, we gave default values for all paramters in the function definition and then separated the parameters dictionary into several subparts
which would need to be optimized individually and manually. Especially the fact that we were unable to come up with an automatic solution and would need to optimize each
parameter set manually convinced us that the HPO networks would not be finished for the BSP and we thus decided to not present their code in the production section.