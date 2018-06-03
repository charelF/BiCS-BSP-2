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




        intervalMin = 10*(50/((100/max(noiseDensity,1))+1))
        intervalMax = 500+intervalMin
        # the intervals are defined in a way such that if our interval is from 0
        # to 100, 50 being the middle, then the noiseDensity percentage is the
        # likelihood of a number being above or below 50

        # ex: noiseDensity of 25 --> interval = 10 - 60, which means that a
        # random value in that interval is 4 times as likely to be below 50 than
        # above 50

        # ex3: noiseDensity of 50 --> interval = 33-66, which means that a
        # random value in that interval is 2 times as likely to be below 50 than
        # above 50

        # ex2: noiseDensity of 100 --> interval = 25-75, which means that a
        # random value in that interval is as likely to be below 50 than
        # above 50

        # we also changed the interval from 0-100 to 0-1000 (and multiplied
        # everything accordingly to have a higher precision when creating
        # datasets)

        randomMatrix = np.random.randint(low=int(intervalMin),
                                         high=int(intervalMax),
                                         size=self.content.shape)
        # we create a randomMatrix with random values in the specified
        # interval and of the same shape as matrix.conent

        randomMatrix = np.where(randomMatrix <= 500, 0, randomMatrix)
        randomMatrix = np.where(randomMatrix > 500, 1, randomMatrix)

        # we make the noise binary (i.e. our randommatrix now contains only
        # 1's and 0's)

        if modifiedLocations == 0:
            # we only add noise to 0's, as we add the randomMatrix to
            # matrix.content and change all values above 1 back to 1
            self.content += randomMatrix
            self.content = np.where(self.content > 1, 1, self.content)

        elif modifiedLocations == 1:
            # we only add noise to 1's, as we subtract the randomMatrix from
            # matrix.content and change all values below 0 back to 0
            self.content -= randomMatrix
            self.content = np.where(self.content < 0, 0, self.content)

        elif modifiedLocations == 2:
            # we add noise to 1's and 0's, as we add the randomMatrix to
            # matrix.content and change all values above 1 to 0
            self.content += randomMatrix
            self.content = np.where(self.content > 1, 0, self.content)



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
                columnSubMatrix : int > 0
                    column size of the matrix SubMatrix.
                rowSubMatrix : int > 0
                    row size of the matrix SubMatrix.

                Returns
                -------
                Does not return, but changes self.content
            """

            columnSubMatrix = min(self.content.shape[0], columnSubMatrix)
            rowSubMatrix = min(self.content.shape[1], rowSubMatrix)
            # dimensions of subMatrix can not exceed dimensions of matrix


            subMatrix = np.ones((columnSubMatrix, rowSubMatrix), dtype=np.int8)
            # the submatrix contains only 1's

            columnCoordinates = random.randint(0, self.content.shape[0] - columnSubMatrix)
            rowCoordinates = random.randint(0, self.content.shape[1] - rowSubMatrix)
            # we choose random Col and Row Coordinates with the limitation that
            # placing our submatrix inside the matrix.content at these coordinates
            # will not cut any parts off the submatrix and it will fit completely inside

            self.content[columnCoordinates:(columnCoordinates + columnSubMatrix),
                         rowCoordinates:(rowCoordinates + rowSubMatrix)] = subMatrix
            # the ":" is use to index the list: a=[x,y,z,v] --> a[0:1] = [x,y]
