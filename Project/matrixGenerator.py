# [B-11 Imports]
import numpy as np   # module used for creating matrices
import random   # module used to find random integers
# import pandas as pd   # module used to export matrix as CSV
# import scipy.misc   # module used to export matrix as image


# [B-12 Matrix Class]
class Matrix:
    def __init__(self, column, row):
        self.content = np.zeros((column, row), dtype=np.int8)
        # we initialize our Matrix class by creating a matrix of type numpy.ndarray,
        # filling it with zeros and assigning the variable content to it.

    # [B-12A addBinaryNoise]
    def addBinaryNoise(self, noiseDensity, modifiedLocations):
        """ The method addBinaryNoise creates a randomMatrix filled with a
            customizable density of 1's and 0's. which is then added or subtracted
            from parts of the main self.content matrix, depending on the choice
            of modifiedLocations.

            Parameters
            ----------
            noiseDensity : int
                A positive integer describing the density of the noise to be added.
                0 means no noise, 50 means noise is added to 50% of the values and
                100 means noise is added to 100% of the values.
                Adding noise to a value means that the value has a 50% chance of
                changing from 0 to 1 or 1 to 0.

            modifiedLocations : int
                Possible arguments: 0, 1, 2.
                0: noise applied to 0's of Matrix
                1: noise applied to 1's of Matrix
                2: noise applied to 0's and 1's of Matrix

            Returns
            -------
            Does not return, but changes self.content

        """


        intervalMin = max(int(0.5 * (noiseDensity / 2)), 0)
        intervalMax = 50 + intervalMin

        randomMatrix = np.random.randint(low=intervalMin,
                                         high=intervalMax,
                                         size=self.content.shape)

        randomMatrix = np.where(randomMatrix < 50, 0, randomMatrix)
        randomMatrix = np.where(randomMatrix >= 50, 1, randomMatrix)

        if modifiedLocations == 0:
            self.content += randomMatrix
            self.content = np.where(self.content > 1, 1, self.content)

        elif modifiedLocations == 1:
            self.content -= randomMatrix
            self.content = np.where(self.content < 0, 0, self.content)

        elif modifiedLocations == 2:
            self.content += randomMatrix
            self.content = np.where(self.content > 1, 0, self.content)

        # return self.content
        # not needed


# [B-13 EmptyMatrix]
class EmptyMatrix(Matrix):
    pass


# [B-14 FeatureMatrix]
class FeatureMatrix(Matrix):
        def fillMatrixWithSubMatrix(self, columnSubMatrix, rowSubMatrix):
            """ The method fillMatrixWithSubMatrix creates a submatrix of size
                columnSubMatrix by rowSubMatrix containing only 1's, as opposed
                to our main matrix which contains only 0's. The subMatrix
                is then placed at a random position inside the main matrix

                Parameters
                ----------
                columnSubMatrix : int
                    column size of the matrix SubMatrix.
                rowSubMatrix : int
                    row size of the matrix SubMatrix.

                Returns
                -------
                Does not return, but changes self.content
            """

            columnSubMatrix = min(self.content.shape[0], columnSubMatrix)
            rowSubMatrix = min(self.content.shape[1], rowSubMatrix)
            # dimensions of subMatrix can not exceed dimensions of matrix


            subMatrix = np.ones((columnSubMatrix, rowSubMatrix), dtype=np.int8)
            columnCoordinates = random.randint(0, self.content.shape[0] - columnSubMatrix)
            rowCoordinates = random.randint(0, self.content.shape[1] - rowSubMatrix)

            self.content[columnCoordinates:(columnCoordinates + columnSubMatrix),
                         rowCoordinates:(rowCoordinates + rowSubMatrix)] = subMatrix
            # the ":" is use to index the list: a=[1,2,3,4] --> a[0:2] = [1,2,3]

            # return self.content

# a=Matrix(30,30)
# a.addBinaryNoise(166567,2)
# print(a.content)
# b=FeatureMatrix(4,4)
# b.fillMatrixWithSubMatrix(3,3)
# print(b.content)

# c=EmptyMatrix(4, 4)
# print(c.content)
# print(c.shape())
