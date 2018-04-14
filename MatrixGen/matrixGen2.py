import numpy as np   # module used for creating matrices
import random   # module used to find random integers
import pandas as pd   # module used to export matrix as CSV
import scipy.misc   # module used to export matrix as image


def fillMwithSM(colM, rowM, colSM, rowSM):
    """ The function generates two matrices, namely the matrix M of size
        colM by rowM, and the submatrix SM of size colSM by rowSM.
        The matrix M contains only 0's and the matrix SM contains only 1's.
        The matrix SM is then placed at a random position inside the matrix M.

        Parameters
        ----------
        colM : int > 0
            column size of the matrix M.
        rowM : int > 0
            row size of the matrix M.
        colSM : int >= 0, int =< colM
            column size of the matrix SM.
        rowSM : int >= 0, int <= colSM
            row size of the matrix SM.

        Returns
        -------
        numpy.ndarray
            Returns a matrix filled with 0's except for a random submatrix
            inside which is filled with 1's.
    """

    M = np.zeros((colM, rowM))  # creates matrix M
    SM = np.full((colSM, rowSM), 1)  # creates matrix A

    colMCord = random.randint(0, (min(colM, rowM))-(colSM))
    # creates random column coordinates in M
    rowMCord = random.randint(0, (min(colM, rowM))-(rowSM))
    # creates random row coordinates in M

    M[colMCord:(colMCord + colSM), rowMCord:(rowMCord + rowSM)] = SM
    # defines the exact position of A inside M
    M = M.astype(int)
    # converts M from containing floats to containing integers

    return M



def addNoise(matrix, noiseDensity, modifiedLocations, noiseType=None):
    """ The function adds different types of noise of different density to
        different parts of a matrix.

        Parameters
        ----------
        matrix : numpy.ndarray
            An input matrix of class numpy.ndarray, which contains only
            0's and 1's.
        noiseDensity : percentage (int between 0 and 100)
            A value describing the density (for binary noise) and quantity (for
            other types of coloring) of noise which is added to the matrix.
        modifiedLocations : 0,1,2
            An integer representing the area where the noise is applied to. 0 means
            the noise is only added to the 0's of the matrix, 1 to the 1s and 2
            means the noise is added to all parts of the matrix.
        noiseType : string, optional
            noiseType defines the type of noise which is added to the matrix.
            default is grayscale, where the noise is different shades of gray
            "binary" means the noise is either black or white
            "color" means the noise is RGB colored
            "colorBinary" means the noise is either max colored or black

        Returns
        -------
        numpy.ndarray
            The function returns the input matrix, but with noise applied to it.
    """

    if noiseType == "color" or noiseType == "colorBinary":
        matrix = np.stack((matrix, matrix, matrix), axis=2)
        # we transform our 2D array of the shape (x,y)
        # into a 3D array of the shape (x,y,z),
        # with z just being our original array 3 times.
        # This process is needed to output a colorful image

    randomMatrix = np.random.random_integers(low=0,
                                             high=noiseDensity,
                                             size=matrix.shape)
    # we create a new matrix of same shape as input matrix
    # and fill it with random integers from 0 to maxIntensity

    randomMatrix = randomMatrix*0.01
    # we transform the integers into floats between 0 and 1

    if modifiedLocations == 0:
        # add randoms to 0s
        combinedMatrix = randomMatrix + matrix
        # we add the random matrix and input matrix
        combinedMatrix = np.where(combinedMatrix >= 1.0, 1.0, combinedMatrix)
        # we replace all entries higher than 1 in the combinedMatrix with 1.
        # As our random matrix only had values between 0 and 1,
        # this steps basically isolates the 1's from our input matrix.

    elif modifiedLocations == 1:
        # subtract randoms from 1s
        combinedMatrix = matrix-randomMatrix
        combinedMatrix = np.where(combinedMatrix < 0.0, 0.0, combinedMatrix)

    else:
        # add randoms to 0s and subtract randoms from 1s
        combinedMatrix = randomMatrix + matrix
        combinedMatrix = np.where(combinedMatrix > 1.0,
                                  2-combinedMatrix,
                                  combinedMatrix)
        """ algorithm explanation:
            1) in a first step, we add randommatrix and matrix, thus
                we get a matrix, where some of the 0's have been replaced
                by 1's and some of the 1's by 2's
            2) so now we need to get rid of the 2's, the way we do that is to
                just find them all and replace them by 0s
            NVM, initial algorithm was overly complicated, easier version
            / comparision:

            test = np.where(self.Matrix > 1.0,
                                   2-self.Matrix,
                                   self.Matrix)

            test2 = np.where(self.Matrix > 1.0,
                                   0,
                                   self.Matrix)

            print(test2==test) --> TRUE
        """

    if noiseType == "binary" or noiseType == "colorBinary":

        combinedMatrix = np.where(combinedMatrix >= 0.5, 1.0, combinedMatrix)
        # all values above 0.5 become white
        combinedMatrix = np.where(combinedMatrix < 0.5, 0.0, combinedMatrix)
        # all values below 0.5 become black

    return combinedMatrix


def dataSetGen(fillMwithSMTuple, addNoiseTuple, dataQuantity,
               fileType, path, namingScheme):
    """ the function dataSetGen creates a desired quantity of different
        matrices, gives them an individual name and then saves them as images
        or text files in a specified location.

    Parameters
    ----------
    fillMwithSMTuple : tuple
        A tuple containing all the parameters of the fillMwithSM function.
        They are (colM, rowM, colSM, rowSM)
    addNoiseTuple : tuple
        A tuple containing all the parameters of the addNoise function excpet
        for the matrix parameter as that one is given by the fillMwithSMTuple.
        They are (noiseDensity, modifiedLocations, noiseType)
    dataQuantity : int
        An integer > 0 representing the amount of elements in the dataset.
    fileType : string
        A string representing the desired datatype of the created file(s)
        Available options are:
            "csv" - stores the data in csv tables
            "jpg" - stores the data in jpg images
            "png" - stores the data in png images
    path : string
        The path of the folder for the generated file(s) to be saved in.
    namingScheme : string
        A name given to the file(s).

    Returns
    -------
    None
        The function returns None, but creates the desired outputs.
    """

    for i in range(dataQuantity):
        matrix = addNoise(fillMwithSM(*fillMwithSMTuple), *addNoiseTuple)
        # we need to create the matrix with tuples as we ned to create it in a
        # loop as it would otherwhise not be possible to make it random

        fileName = path + namingScheme + str(i) + "." + fileType
        # the name of the final file is a combination of a string given by
        # the user and an index given by the for loop

        if fileType == "csv":
            # we use a built-in pandas method to create csv tables
            table = pd.DataFrame(matrix)
            table.to_csv(fileName, header=None, index=None)

        elif fileType == "jpg" or "png":
            # we use a built-in scipy method to create images
            scipy.misc.imsave(fileName, matrix)


def dataFileGen(fillMwithSMTuple, addNoiseTuple, dataQuantity,
                fileType, path, namingScheme):
    """ the function dataFileGen appends random matrices to one big matrix and
        then allows to save that big matrix in a csv table, npy file or return
        it as an output of the function.

    Parameters
    ----------
    fillMwithSMTuple : tuple
        A tuple containing all the parameters of the fillMwithSM function.
        They are (colM, rowM, colSM, rowSM)
    addNoiseTuple : tuple
        A tuple containing all the parameters of the addNoise function excpet
        for the matrix parameter as that one is given by the fillMwithSMTuple.
        They are (noiseDensity, modifiedLocations, noiseType)
    dataQuantity : int
        An integer > 0 representing the amount of elements in the dataset.
    fileType : string
        A string representing the desired datatype of the created file(s)
        Available options are:
            "csv" - stores the data in a csv table
            "npy" - stores the data in a npy file
            "variable" - returns the function as variable of type numpy.ndarray
    path : string
        The path of the folder for the generated file(s) to be saved in.
    namingScheme : string
        A name given to the file.

    Returns
    -------
    numpy.ndarray | None
        The return of the function depends on the fileType. The function
        returns None if filetype is csv or npy and returns a matrix of type
        numpy.ndarray if the filetype is a variable.

    """

    matrixList = []  # this local variable stores the list

    for i in range(dataQuantity):
        matrix = addNoise(fillMwithSM(*fillMwithSMTuple), *addNoiseTuple)
        # we need to create the matrix with tuples as we ned to create it in a
        # loop as it would otherwhise not be possible to make it random

        matrixList.append(matrix)
        # we append the generated matrices to the list

        matrixNpArray = np.array(matrixList)
        # we transform our matrix of type list into a matrix of type ndarray to
        # be able to be able to profit from the more efficient and improved
        # handling of arrays by numpy compared to standard python lists.

    if fileType == "csv":
        #!!!!! CURRENTLY NOT WORKING DUE TO INCREASED DIMENSION!!!!!
        # we use a built-in pandas method to create csv tables

        fileName = path + namingScheme + "." + fileType

        table = pd.DataFrame(matrixNpArray)
        table.to_csv(fileName, header=None, index=None)
        # this is similar to the csv implementation of the dataSetGen, but
        # instead of using multiple tables we use only one.

    elif fileType == "npy":
        # we use the filetype .npy to store the matrix. This filetype is very
        # efficient and optimized for storing numpy.ndarrays

        fileName = path + namingScheme
        np.save(fileName, matrixNpArray)
        # to open this in the future: np.load("fileName.npy")

    elif fileType == "variable":
        # if the desired filetype is a variable, we simply return the function
        return matrixNpArray

"""
dataFileGen((64, 64, 5, 5),
                (60,
                 0,
                 "binary"),
                dataQuantity=2000,
                fileType="csv",
                path="D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\",
                namingScheme="matrix")

"""
"""
dataSetGen((64, 64, 5, 5),
                (60,
                 0,
                 "binary"),
                dataQuantity=20,
                fileType="csv",
                path="D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\mini_dataset_csv",
                namingScheme="matrix")
"""
"""
print(len(a))
print(type(a))
print(type(a[4]))
print(a[4].shape)
print(a[4].size)
print(a[4][4], a[10][4], a[13][4])
"""
