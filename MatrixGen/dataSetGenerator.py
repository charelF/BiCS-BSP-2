from matrixGenerator import *

"""
a = FeatureMatrix(8, 9)
print(a.content)
a.fillMatrixWithSubmatrix(3, 4)
print(a.content)
a.addBinaryNoise(33, 0)
print(a.content, type(a.content), a.content.shape, a.content.size)
"""

def saveDataset(filename, dataset):
    """ Inspired by a similar application from Benjamin Jahic
    """
    with open(filename, "w") as file:
        file.write("# Array info: {}\n".format(dataset.shape))
        count = 1

        for array in dataset:
            file.write("#________________Entry__Number__{}________\n".format(count))
            np.savetxt(file, array, fmt="%0u")
            count += 1

#saveDataset("test2.txt", a.content)


def createDataset(amount, filename, MatrixParam, SubmatrixParam, NoiseParam=None):
    outputFileName = filename + "_output" + ".txt"
    inputFileName = filename + "_input" + ".txt"

    # Generation of input set
    inputSet=[]
    for i in range(amount):

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
    outputSet=[i % 2 for i in range(amount)]
    outputSet = np.array(outputSet)
    outputSet = np.reshape(outputSet, (200, 1, 1))
    saveDataset(outputFileName, outputSet)

createDataset(200, "test", (12, 12), (6, 6), (33, 0))
