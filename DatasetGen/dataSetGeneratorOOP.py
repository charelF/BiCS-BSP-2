# decide whether to write dataset or dataSet and generalize this in program and report
import os
os.chdir("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
# somehow the files think they are in another directory than they really are.

from matrixGenerator import *
import re  # for the loadDataset

class Dataset:
    def __init__(self, size, matrixParam, subMatrixParam, noiseParam):
        self.size = size

    def create():
        # algorithm to create inputSet
        # algorithm to create outputSet
        return inputSet, outputSet

    def save(dataset, filename, description):

    def load(filename):

class InputSet(Dataset):
    def __init__(matrixParam, subMatrixParam, noiseParam):
        self.matrixParam = matrixParam
        self.subMatrixParam = subMatrixParam
        self.noiseParam = noiseParam


class OutputSet(Dataset):
    pass



class Dataset:
    def __init__(matrixParam, subMatrixParam, noiseParam):
        self.matrixParam = matrixParam
        self.subMatrixParam = subMatrixParam
        self.noiseParam = noiseParam

    def create(size):
        inputSet=[]
        for i in range(size):
            if i % 2:
                temporaryMatrix = FeatureMatrix(*self.matrixParam)
                temporaryMatrix.fillMatrixWithSubMatrix(*self.subMatrixParam)
            else:
                temporaryMatrix = EmptyMatrix(*self.matrixParam)
            temporaryMatrix.addBinaryNoise(*self.noiseParam)
            inputSet.append(temporaryMatrix.content)
        inputSet = np.array(inputSet)

        outputSet=[i % 2 for i in range(size)]
        outputSet = np.array(outputSet)
        outputSet = outputSet.reshape(size, 1, 1)

        return inputSet, outputSet


def save(dataset, filename, description):
    with open(filename, "w") as file:
        file.write("## Dataset Dimensions: {}\n".format(dataset.shape))
        file.write("# Filename: {}\n".format(filename))
        file.write("# Description: {}\n".format(description))
        file.write("# MatrixParam: {}\n".format(self.matrixParam))
        file.write("# SubMatrixParam: {}\n".format(self.subMatrixParam))
        file.write("# NoiseParam: {}##\n".format(self.noiseParam))

        count = 1
        for matrix in dataset:
            file.write("# Entry number: {}\n".format(count))
            np.savetxt(file, matrix, fmt="%1u")
            count += 1


def load(filename):
    # description
    text = open(filename).read()

    print("\n\n" + re.search("##.*##", text, flags=re.DOTALL).group(0) + "\n\n")
    # this prints the description and additional information
    # the DOTALL flag of the re is used to find multiline regular ex. hits
    # our regular expression searches for anything between two hashes

    # dataset
    dataset = np.loadtxt(filename)

    dimensionTupleFinder = re.search("[(]\d+, \d+, \d+[)]", text).group(0)
    # dimensionTupleFinder finds all tuples containing 3 decimals

    dimensionValueFinder = re.findall("\d+", dimensionTupleFinder)
    # dimensionValueFinder finds all integers, as we give it the previous re,
    # it gives us a list of strings, each string contains a decimal

    dimensions = tuple(int(i) for i in valueFinder)
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
