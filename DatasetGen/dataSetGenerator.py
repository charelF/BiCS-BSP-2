import os
os.chdir("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
# somehow the files think they are in another directory than they really are.

from matrixGenerator import *
import re  # for the loadDataset


def createDataset(size, filename, MatrixParam, SubmatrixParam, NoiseParam=None):
    outputFileName = filename + "_output" + ".txt"
    inputFileName = filename + "_input" + ".txt"

    # Generation of input set
    inputSet=[]
    for i in range(size):

        if i % 2:
            temporaryMatrix = FeatureMatrix(*MatrixParam)
            temporaryMatrix.fillMatrixWithSubmatrix(*SubmatrixParam)
        else:
            temporaryMatrix = EmptyMatrix(*MatrixParam)

        if NoiseParam:
            temporaryMatrix.addBinaryNoise(*NoiseParam)

        inputSet.append(temporaryMatrix.content)

    inputSet = np.array(inputSet)
    # we don't need to reshape as it is already in the correct shape
    saveDataset(inputFileName, inputSet)


    # Generation of output set
    outputSet=[i % 2 for i in range(size)]
    outputSet = np.array(outputSet)
    outputSet = outputSet.reshape(size, 1, 1)
    saveDataset(outputFileName, outputSet)


def saveDataset(filename, dataset):
    """ Inspired by a similar application from Benjamin Jahic
    """
    with open(filename, "w") as file:
        file.write("#________________Array__Info:__{}__________\n".format(dataset.shape))
        count = 1

        for array in dataset:
            file.write("#________________Entry__Number__{}________\n".format(count))
            np.savetxt(file, array, fmt="%0u")
            count += 1


def loadDataset(filename):
    dataset = np.loadtxt(filename)

    text = open(filename).read()
    regEx = re.search("(?:\()\d+,.\d+,.\d+(?:\))", text).group(0)
    # regEx is all tuples with 3 decimals found in the text
    valueFinder = re.findall("\d+", regEx)
    # valueFinder is a list of all decimals in regEx
    dimensions = tuple(int(i) for i in valueFinder)

    dataset = dataset.reshape((*dimensions))
    return dataset

