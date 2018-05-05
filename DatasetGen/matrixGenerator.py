import os
os.chdir("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
# somehow the files think they are in another directory than they really are.

import numpy as np   # module used for creating matrices
import random   # module used to find random integers
# import pandas as pd   # module used to export matrix as CSV
# import scipy.misc   # module used to export matrix as image


class Matrix:
    def __init__(self, column, row):
        self.column = column
        self.row = row
        self.content = np.zeros((column, row), dtype=np.int8)

    def addBinaryNoise(self, noiseDensity, modifiedLocations):
        """ The function addBinaryNoise ads and subtracts random values from
            the matrix, but does not add new values: It replaces certain 1's
            with 0's and vice versa, but does not add intermediary values.

            Parameters
            ----------
            noiseDensity : int
                Description of parameter `noiseDensity`.
            modifiedLocations : type
                Description of parameter `modifiedLocations`.

            Returns
            -------
            type
                Description of returned object.
        """

        noiseDensity = (noiseDensity//2)+50
        randomMatrix = np.random.randint(low=0, high=noiseDensity, size=self.content.shape)

        randomMatrix = np.where(randomMatrix < 50, 0, randomMatrix)
        randomMatrix = np.where(randomMatrix >= 50, 1, randomMatrix)

        if modifiedLocations == 0:
            self.content += randomMatrix
            self.content = np.where(self.content > 1, 1, self.content)
            return self.content

        elif modifiedLocations == 1:
            self.content -= randomMatrix
            self.content = np.where(self.content < 0, 0, self.content)
            return self.content

        else:
            self.content += randomMatrix
            self.content = np.where(self.content > 1, 0, self.content)
            return self.content


class EmptyMatrix(Matrix):
    pass

class FeatureMatrix(Matrix):
        def fillMatrixWithSubmatrix(self, columnSubmatrix, rowSubmatrix):
            """ The function generates two matrices, namely the matrix M of size
                column by row, and a submatrix of size columnSubmatrix by
                rowSubmatrix. The matrix contains only 0's and the submatrix
                contains only 1's. The submatrix is then placed at a random
                position inside the matrix.

                Parameters
                ----------
                column : int > 0
                    column size of the matrix M.
                row : int > 0
                    row size of the matrix M.
                columnSubmatrix : int >= 0, int =< column
                    column size of the matrix submatrix.
                rowSubmatrix : int >= 0, int <= columnSubmatrix
                    row size of the matrix submatrix.

                Returns
                -------
                numpy.ndarray
                    Returns a matrix filled with 0's except for a random submatrix
                    inside which is filled with 1's.
            """
            temp = [columnSubmatrix, rowSubmatrix]
            for i in range(len(self.content.shape)):
                if self.content.shape[i] < temp[i]:
                    temp[i] = self.content.shape[i]
            columnSubmatrix, rowSubmatrix = temp
            # this removes the error from submatrix being larger than matrixself.
            # the code needs to be optimised, and simplified


            submatrix = np.ones((columnSubmatrix, rowSubmatrix), dtype=np.int8)
            columnCoordinates = random.randint(0, (min(self.column, self.row))-(columnSubmatrix))
            rowCoordinates = random.randint(0, (min(self.column, self.row))-(rowSubmatrix))

            self.content[columnCoordinates:(columnCoordinates + columnSubmatrix),
                         rowCoordinates:(rowCoordinates + rowSubmatrix)] = submatrix
            # quick note, the ":" is not used like for ex in Swift, but just to
            # index the list, like in python: a=[1,2,3,4] --> a[0:2] = [1,2,3]

            return self.content
