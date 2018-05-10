import os
os.chdir("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
# somehow the files think they are in another directory than they really are.

import numpy as np   # module used for creating matrices
import random   # module used to find random integers
# import pandas as pd   # module used to export matrix as CSV
# import scipy.misc   # module used to export matrix as image


class Matrix:
    def __init__(self, column, row):
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
        def fillMatrixWithSubMatrix(self, columnSubMatrix, rowSubMatrix):
            """ The function generates two matrices, namely the matrix M of size
                column by row, and a SubMatrix of size columnSubMatrix by
                rowSubMatrix. The matrix contains only 0's and the SubMatrix
                contains only 1's. The SubMatrix is then placed at a random
                position inside the matrix.

                Parameters
                ----------
                column : int > 0
                    column size of the matrix M.
                row : int > 0
                    row size of the matrix M.
                columnSubMatrix : int >= 0, int =< column
                    column size of the matrix SubMatrix.
                rowSubMatrix : int >= 0, int <= columnSubMatrix
                    row size of the matrix SubMatrix.

                Returns
                -------
                numpy.ndarray
                    Returns a matrix filled with 0's except for a random SubMatrix
                    inside which is filled with 1's.
            """

            columnSubMatrix = min(self.content.shape[0], columnSubMatrix)
            rowSubMatrix = min(self.content.shape[1], rowSubMatrix)
            # dimensions of subMatrix can not exceed dimensions of matrix


            SubMatrix = np.ones((columnSubMatrix, rowSubMatrix), dtype=np.int8)
            columnCoordinates = random.randint(0, self.content.shape[0] - columnSubMatrix)
            rowCoordinates = random.randint(0, self.content.shape[1] - rowSubMatrix)

            self.content[columnCoordinates:(columnCoordinates + columnSubMatrix),
                         rowCoordinates:(rowCoordinates + rowSubMatrix)] = SubMatrix
            # the ":" is use to index the list: a=[1,2,3,4] --> a[0:2] = [1,2,3]

            return self.content
