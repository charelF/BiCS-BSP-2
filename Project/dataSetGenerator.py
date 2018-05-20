# [B-21 Imports]
from matrixGenerator import *
import re  # for the loadDataset


# [B-22 createDataset]
def createDataset(size, filename, description, matrixParam, subMatrixParam, noiseParam):
    outputFileName = filename + "_output" + ".txt"
    inputFileName = filename + "_input" + ".txt"

    # [B-22A creation of inputSet]
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
    saveDataset(inputSet, inputFileName, description, matrixParam, subMatrixParam, noiseParam)


    # [B-22B creation of outputSet]
    # Generation of output set
    outputSet=[i % 2 for i in range(size)]
    # print(outputSet) --> [0, 1, 0, 1, 0, 1, 0, ...] class: list
    outputSet = np.array(outputSet)
    # print(outputSet) --> [0 1 0 1 0 1 ...] class: nump.ndarray
    # print(outputSet.shape) --> (size,)
    outputSet = outputSet.reshape(size, 1, 1)
    # print(outputSet) --> [[[0]] [[1]] [[0] [[1]] ...]
    # print(outputSet.shape) --> (size,1,1), ex: (2000, 1, 1)
    saveDataset(outputSet, outputFileName, description, matrixParam, subMatrixParam, noiseParam)


# [B-23 saveDataset]
def saveDataset(dataset, filename, description, matrixParam, subMatrixParam, noiseParam):
    """ Inspired by a similar application from Benjamin Jahic
    """
    with open(filename, "w") as file:
        file.write("## Dataset Dimensions: {}\n".format(dataset.shape))
        file.write("# Filename: {}\n".format(filename))
        file.write("# Description: {}\n".format(description))
        file.write("# MatrixParam: {}\n".format(matrixParam))
        file.write("# SubMatrixParam: {}\n".format(subMatrixParam))
        file.write("# NoiseParam: {}##\n".format(noiseParam))

        count = 1
        for matrix in dataset:
            file.write("# Entry number: {}\n".format(count))
            np.savetxt(file, matrix, fmt="%1u")
            count += 1


# [B-24 loadDataset]
def loadDataset(filename):
    # description
    text = open(filename).read()

    # [B-24A print description]
    print("\n\n" + re.search("##.*##", text, flags=re.DOTALL).group(0) + "\n\n")
    # this prints the description and additional information
    # the DOTALL flag of the re is used to find multiline regular ex. hits
    # our regular expression searches for anything between two hashes

    # [B-24B return dimension tuple]
    # dataset
    dataset = np.loadtxt(filename)

    dimensionTupleFinder = re.search("[(]\d+, \d+, \d+[)]", text).group(0)
    # dimensionTupleFinder finds all tuples containing 3 decimals

    dimensionValueFinder = re.findall("\d+", dimensionTupleFinder)
    # dimensionValueFinder finds all integers, as we give it the previous re,
    # it gives us a list of strings, each string contains a decimal

    dimensions = tuple(int(i) for i in dimensionValueFinder)
    # We convert the strings of the previous list into integers using list
    # comprehension. We have to declare the final object an int, as it
    # otherwhise is a generator object

    dataset = dataset.reshape(dimensions)
    # we can finally reshape our dataset to the dimensions it was saved it

    return dataset

# def loadDatasetDescription(filename):
#     text = open(filename).read()
#     regEx = re.search("##.*##", text, flags=re.DOTALL).group(0)
#     return regEx
