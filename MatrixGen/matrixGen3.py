import numpy as np   # module used for creating matrices
import random   # module used to find random integers
import pandas as pd   # module used to export matrix as CSV
import scipy.misc   # module used to export matrix as image


class Matrix:
    def __init__(self, column, row):
        self.column = column
        self.row = row
        self.content = np.zeros((column, row), dtype=np.int8)

    def addBinaryNoise(self, noiseDensity, modifiedLocations):
        randomMatrix = np.random.randint(low=0,
                                         high=noiseDensity,
                                         size=self.content.shape)

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
            submatrix = np.ones((columnSubmatrix, rowSubmatrix), dtype=np.int8)
            columnCoordinates = random.randint(0, (min(self.column, self.row))-(columnSubmatrix))
            rowCoordinates = random.randint(0, (min(self.column, self.row))-(rowSubmatrix))

            self.content[columnCoordinates:(columnCoordinates + columnSubmatrix),
                         rowCoordinates:(rowCoordinates + rowSubmatrix)] = submatrix

            return self.content


a = FeatureMatrix(8, 9)
print(a.content)
a.fillMatrixWithSubmatrix(3, 4)
print(a.content)





"""
dataset=[]

for i in range(2000):
    WS = Matrix(32, 32)
    WOS = Matrix(32, 32)
    WS.fillMwithsubmatrix(7, 7)
    WS.addBinaryNoise(60, 2)
    WOS.addBinaryNoise(60, 2)
    WS.flatten()
    WOS.flatten()
    WS.OutputClassification()
    WOS.OutputClassification()
    dataset.append(WS.matrix)
    dataset.append(WOS.matrix)


table = pd.DataFrame(dataset)
table.to_csv("test.csv", header=None, index=None)


"""


"""
def go():
    test = Matrix(32, 32)
    test.fillMwithsubmatrix(5, 5)
    test.flatten()
    print(test.containsFeature, test.matrixDim)
    print(test.matrix.shape)
    print(test.matrix[::-1])
    test.OutputClassification()
    print(test.matrix[::-1])
    print(test.matrix.shape)
    # Everything works perfectly as expected

    test.fillMwithsubmatrix(7, 7)
    test.addBinaryNoise(55, 0)
    scipy.misc.imsave("testi.png", test.matrix)

"""





"""
def flatten(self):
    # flattens the matrix to be used in the ANN
    self.content = np.reshape(self.content, self.content.size)
    self.contentDim = 1
    return self.content

def OutputClassification(self):
    if self.contentDim == 1:
        if self.containsFeature:
            self.content = np.append(self.content, [1])
            # we add a 1 to the end to signal
            # that this matrix contains a feature
        else:
            self.content = np.append(self.content, [0])
        # here comes the code we will use to add the classification
        # feature to our 2D array, i.e. how we will separate a 2D array
        # with and without the square of 1's
    return self.content
"""











"""

still does not work. The goal is to have a dataset of the same shape as
the one in the udemy ANN example
to do this, look into pandas writing (by rows) methods


scipy.misc.imsave("testi.png", a)




matrixList = np.empty(shape=(1, 25), dtype=np.int8)
print(matrixList, matrixList.shape)

np.append(matrixList, m.matrix)

print(matrixList)

"""
"""
table = pd.DataFrame(m.matrix)
table.to_csv("fileName", header=None, index=None)

"""
"""
    matrixList = []

    for i in range(100):
        m = Matrix(32, 32)
        m.fillMwithsubmatrix(5, 5)
        m.addBinaryNoise(90, 2)
        matrixList.append(m.matrix)

    matrixNpArray = np.array(matrixList)
    np.save("test", matrixNpArray)
    print(matrixNpArray[4].shape, matrixNpArray.shape)
    np.load("test.npy")

    scipy.misc.imsave("testi.png", matrixNpArray[4])
"""
