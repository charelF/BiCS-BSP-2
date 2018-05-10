# decide whether to write dataset or dataSet and generalize this in program and report
import os
os.chdir("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
# somehow the files think they are in another directory than they really are.

from matrixGenerator import *
import re  # for the loadDataset


def createDataset(size, filename, description, matrixParam, subMatrixParam, noiseParam):
    outputFileName = filename + "_output" + ".txt"
    inputFileName = filename + "_input" + ".txt"

    # Generation of input set
    inputSet=[]
    for i in range(size):
        if i % 2:
            temporaryMatrix = FeatureMatrix(*matrixParam)
            temporaryMatrix.fillMatrixWithSubMatrix(*subMatrixParam)
        else:
            temporaryMatrix = EmptyMatrix(*matrixParam)

        if noiseParam:
            temporaryMatrix.addBinaryNoise(*noiseParam)

        inputSet.append(temporaryMatrix.content)

    inputSet = np.array(inputSet)
    # print(inputSet) --> [[[000][010]...] [[111]...]] ...] class: numpy.ndarray
    # print(inputSet.shape) --> (size, column, row), ex: (2000, 32, 32)
    saveDataset(inputFileName, inputSet, description)


    # Generation of output set
    outputSet=[i % 2 for i in range(size)]
    # print(outputSet) --> [0, 1, 0, 1, 0, 1, 0, ...] class: list
    outputSet = np.array(outputSet)
    # print(outputSet) --> [0 1 0 1 0 1 ...] class: nump.ndarray
    # print(outputSet.shape) --> (size,)
    outputSet = outputSet.reshape(size, 1, 1)
    # print(outputSet) --> [[[0]] [[1]] [[0] [[1]] ...]
    # print(outputSet.shape) --> (size,1,1), ex: (2000, 1, 1)
    saveDataset(outputFileName, outputSet, description)


def saveDataset(filename, dataset, description):
    """ Inspired by a similar application from Benjamin Jahic
    """
    with open(filename, "w") as file:
        file.write("#________________Array info: {}\n".format(dataset.shape))
        file.write("#________________Description: {}\n".format(description))
        count = 1

        for array in dataset:
            file.write("#________________Entry number: {}\n".format(count))
            np.savetxt(file, array, fmt="%0u")
            count += 1


def loadDataset(filename):
    dataset = np.loadtxt(filename)

    text = open(filename).read()
    regEx = re.search("[(]\d+, \d+, \d+[)]", text).group(0)
    # regEx is all tuples with 3 decimals found in the text
    valueFinder = re.findall("\d+", regEx)
    # valueFinder is a list of all decimals in regEx
    dimensions = tuple(int(i) for i in valueFinder)

    dataset = dataset.reshape((*dimensions))
    return dataset

def loadDatasetDescription(filename):
    text = open(filename).read()
    regEx = re.search("Description: .*", text).group(0)
    return regEx
