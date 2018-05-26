# [B-21 Imports]
import MatrixGenerator as mg
import numpy as np
import re  # for the loadDataset


# [B-22 createDataset]
def createDataset(size, filename, description, matrixParam, subMatrixParam, noiseParam):

    # [B-22A inputSet]
    inputFileName = filename + "_input" + ".txt"
    with open(inputFileName, "w") as file:
        # this creates or opens a file called inputFileName, sets it to write
        # mode, and assigns the variable file to it

        # we write the dimension in the form (size, col, row); this is
        # necessary, so the loadDataset can find out the shape of the dataaset
        file.write("## Dataset Dimensions: {}\n".format(tuple((size, *matrixParam))))

        # we also write some unecessary but helpful information
        file.write("# Filename: {}\n".format(filename))
        file.write("# Description: {}\n".format(description))
        file.write("# MatrixParam: {}\n".format(matrixParam))
        file.write("# SubMatrixParam: {}\n".format(subMatrixParam))
        file.write("# NoiseParam: {}##\n".format(noiseParam))

        for i in range(size):
            if i % 2:
                temporaryMatrix = mg.FeatureMatrix(*matrixParam)
                temporaryMatrix.fillMatrixWithSubMatrix(*subMatrixParam)
            else:
                temporaryMatrix = mg.EmptyMatrix(*matrixParam)

            temporaryMatrix.addBinaryNoise(*noiseParam)

            file.write("# Entry number: {}\n".format(i))
            np.savetxt(file, temporaryMatrix.content, fmt="%1u")


    # [B-22B outputset]
    outputFileName = filename + "_output" + ".txt"
    with open(outputFileName, "w") as file:
        file.write("## Dataset Dimensions: {}\n".format(tuple((size, 1, 1))))
        file.write("# Filename: {}\n".format(filename))
        file.write("# Description: {}\n".format(description))
        file.write("# MatrixParam: {}\n".format(matrixParam))
        file.write("# SubMatrixParam: {}\n".format(subMatrixParam))
        file.write("# NoiseParam: {}##\n".format(noiseParam))

        for i in range(size):
            file.write("# Entry number: {}\n".format(i))
            np.savetxt(file, np.array([i%2]) , fmt="%1u")



# [B-23 loadDataset]
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
