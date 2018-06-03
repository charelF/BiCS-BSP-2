# [B-21 Imports]
import MatrixGenerator as mg
import numpy as np
import re  # for the loadDataset


# [B-22 createDataset]
def createDataset(size, filename, description, matrixParam, subMatrixParam, noiseParam):
    """ The function createDataset creates a Dataset containing size amount of elements of
        type numpy.ndarray which have been defined using the matrix, subMatrix and
        noiseParam. The dataset is created as a .txt file called filename and containing
        as comments the arguments of createDataset and a small description.

        Parameters
        ----------
        size : int > 0
            amount of elements that should be in the dataset
        filename : string
            name to give to the file, is followed by "_input.txt" resp. "_output.txt"
        description : string
            string to contain in dataset for user to read
        matrixParam : tuple of 2 ints respecting matrix class initialization values
            arguments used to create Matrix object
        subMatrixParam : tuple of 2 ints respecting fillMatrixWithSubMatrix method
            arguments used to apply fillMatrixWithSubMatrix on Matrix object
        noiseParam : tuple of 2 ints respecting addBinaryNoise method
            arguments used to apply addBinaryNoise on Matrix object

        Returns
        -------
        no return, creates 2 .txt datasets

    """

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
            # the above conditional statement creates alternatively an Empty and a
            # FeatureMatrix, as for every second i, i%2=1, which equals to True

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
            # we save an array containing alternatively 1 or 0 to our output
            # dataset



# [B-23 loadDataset]
def loadDataset(filename):
    """ The function loadDataset loads a given dataset if it is a .txt file and was
        created by createDataset or have the same structure. If succesfully loaded,
        loadDataset extracts the dimension tuple from the description of the datasets
        and reshapes the dataset to the correct dimension before returning it.

        Parameters
        ----------
        filename : string
            Name of the file to load

        Returns
        -------
        numpy.ndarray
            Returns reshaped dataset extracted from the file

    """

    # description
    text = open(filename).read()

    print("\n\n" + re.search("##.*##", text, flags=re.DOTALL).group(0) + "\n\n")
    # this prints the description and additional information
    # the DOTALL flag of the re is used to find multiline regular ex. hits
    # our regular expression searches for anything between two hashes


    # dataset
    dataset = np.loadtxt(filename)

    dimensionTupleFinder = re.search("[(]\d+, \d+, \d+[)]", text)
    # dimensionTupleFinder finds all tuples containing 3 decimals

    numberInTupleFinder = re.findall("\d+", dimensionTupleFinder.group(0))
    # numberInTupleFinder finds all integers, as we give it the previous re,
    # it gives us a list of strings, each string contains a decimal

    dimensions = tuple(int(number) for number in numberInTupleFinder)
    # We convert the strings of the previous list into integers using list
    # comprehension. We have to declare the final object an int, as it
    # otherwhise is a generator object

    dataset = dataset.reshape(dimensions)
    # we can finally reshape our dataset to the dimensions it was saved it

    return dataset
