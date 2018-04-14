import numpy as np
import csv #not sure if needed
import random
import pandas as pd
import matplotlib.pyplot as plt
#import scipy


def fillMwithA(colM, rowM, colA, rowA):
    """ colM and rowM create an 0 matrix, colA and rowA create a 1 matrix
        outputs a matrix of size colM by rowM with a matrix of size colA by rowA randomly inside"""

    M=np.zeros((colM, rowM))

    A=np.full((colA, rowA), 1)

    colMCord=random.randint(0, (min(colM, rowM))-(colA))

    rowMCord=random.randint(0, (min(colM, rowM))-(rowA))

    M[colMCord:(colMCord+colA), rowMCord:(rowMCord+rowA)]=A

    M=M.astype(int)

    return M

def matrixToFile(amount, filetype, colM, rowM, colA, rowA, path=""):
    """ accepts amount(int) of files, filetype(str)(csv or jpg), path(str) (relative) and dimensions(int) of array and matrix inside
        creates the requested files in the requested directory"""

    if filetype=="csv":
        for i in range(amount):
            matrix=fillMwithA(colM, rowM, colA, rowA)
            tableName=path+'matrix'+str(i)+'.csv'
            table=pd.DataFrame(matrix)
            table.to_csv(tableName, header=None, index=None)

    else:
        for i in range(amount):
            matrix=fillMwithA(colM, rowM, colA, rowA)
            imageName=path+'matrix'+str(i)+'.jpg'
            plt.imsave(imageName, matrix)


##matrixToFile(8000, "jpg", 64, 64, 3, 3,) #cant get path to work
#matrixToFile(20, "jpg", "\withoutSquare", 32, 32, 3, 3)





#unused-----------------
#scipy.misc.imsave("test.jpg", matrix)

#image=Image.fromarray(matrix)
#image.save('test.jpg')

            """ unused code from matrixgen2
    Parameters:
    nom + intention

    Description:
    describe the function main computation goal (state modifications, message sending)

    UNUSED---------------------------------------------------

    if filetype=="csv":
        for i in range(amount):
            matrix=fillMwithA(colM, rowM, colA, rowA)
            tableName=path+'matrix'+str(i)+'.csv'
            table=pd.DataFrame(matrix)
            table.to_csv(tableName, header=None, index=None)

    else:
        for i in range(amount):
            matrix=fillMwithA(colM, rowM, colA, rowA)
            imageName=path+'matrix'+str(i)+'.jpg'
            plt.imsave(imageName, matrix)

            #unused-----------------
#scipy.misc.imsave("test.jpg", matrix)

#image=Image.fromarray(matrix)
#image.save('test.jpg')


            import matplotlib.pyplot as plt #module used to export matrix as JPG
            #imageName=path+'matrix'+str(i)+'.jpg'
            #plt.imsave(imageName, matrix)
            #print("dataset succesfully created")
"""

"""
def addColorNoise(matrix, amount, noisetype):

    Parameters:
    matrix: a matrix of any size and of type np.array (created with numpy)
    amount: a percentage value representing the amount of noise to be added to the matrix
    noiseType: an integer describing the part of the matrix the noise is added to
               (0 means noise is applied to 0s, 1 to 1s and 2 to both)

    Description:
    Adds a specific amount and type of RGB colored noise to the matrix


    matrix=np.stack((matrix, matrix, matrix), axis=2) #we transform our 2D array of the shape (x,y) into a 3D array of the shape (x,y,z),
    #with z just being our original array 3 times. This process is needed to output a colorful image

    randomMatrix = np.random.random_integers(0, high=amount, size=matrix.shape)
    #we create a new matrix of same shape as input matrix and fill it with random integers from 0 to maxIntensity
    #print(randomMatrix.shape)

    randomMatrix = randomMatrix*0.01 #we transform the integers into floats between 0 and 1

    if noisetype==0:
        #add randoms to 0s
        combinedMatrix = randomMatrix+matrix #we add the random matrix and input matrix
        combinedMatrix = np.where(combinedMatrix >= 1.0, 1.0, combinedMatrix)
        #we replace all entries higher than 1 in the combinedMatrix with 1. As our random matrix only had values between 0 and 1,
        #this steps basically isolates the 1's from our input matrix.

    elif noisetype==1:
        #subtract randoms from 1s
        combinedMatrix = matrix-randomMatrix
        combinedMatrix = np.where(combinedMatrix < 0.0, 0.0, combinedMatrix)

    else:
        #add randoms to 0s and subtract randoms from 1s
        combinedMatrix = randomMatrix+matrix
        combinedMatrix = np.where(combinedMatrix > 1.0, 1-(combinedMatrix%1.0), combinedMatrix)

    return combinedMatrix


def addBinaryNoise(matrix, amount, noisetype):

    Parameters:
    matrix: a matrix of any size and of type np.array (created with numpy)
    amount: a percentage value representing the amount of noise to be added to the matrix
    noiseType: an integer describing the part of the matrix the noise is added to
               (0 means noise is applied to 0s, 1 to 1s and 2 to both)

    Description:
    Adds a specific amount and type of binary (black or white) noise to the matrix


    randomMatrix = np.random.random_integers(0, high=amount, size=matrix.shape)
    #we create a new matrix of same size as input matrix and fill it with random integers from 0 to maxIntensity

    randomMatrix = randomMatrix*0.01 #we transform the integers into floats between 0 and 1

    if noisetype==0:
        #add randoms to 0s
        combinedMatrix = randomMatrix+matrix #we add the random matrix and input matrix

        combinedMatrix = np.where(combinedMatrix >= 1.0, 1.0, combinedMatrix)
        #we replace all entries higher than 1 in the combinedMatrix with 1. As our random matrix only had values between 0 and 1,
        #this steps basically isolates the 1's from our input matrix.

    elif noisetype==1:
        #subtract randoms from 1s
        combinedMatrix = matrix-randomMatrix
        combinedMatrix = np.where(combinedMatrix < 0.0, 0.0, combinedMatrix)

    else:
        #add randoms to 0s and subtract randoms from 1s
        combinedMatrix = randomMatrix+matrix
        combinedMatrix = np.where(combinedMatrix > 1.0, 1-(combinedMatrix%1.0), combinedMatrix)

    combinedMatrix = np.where(combinedMatrix >= 0.5, 1.0, combinedMatrix) #all values above 0.5 become white
    combinedMatrix = np.where(combinedMatrix < 0.5, 0.0, combinedMatrix) #all values below 0.5 become black

    return combinedMatrix

"""

def dataSetGenOLD(quantity, size, fileType, mod1 = None, mod2 = None, mod3 = None, path="", name="matrix", modifier = None):
    """
    Parameters:
    quantity: a positive integer representing the quantity of elements in the dataset
    size: 4-tuple of the form (colM, rowM, colSM, rowSM)
          colM: a positive integer of the columns quantity of the matrix M of the FillMwithSM function
          rowM: a positive integer of the rows quantity of the matrix M of the FillMwithSM function
          colSM: a positive integer of the rows quantity of the matrix A of the FillMwithSM function
          rowSM: a positive integer of the rows quantity of the matrix A of the FillMwithSM function
    fileType: string being either "jpg" or "csv" or "png" and representing the fileType of the dataset
    mod1, mod2: optional modifier arguments for the modifier function, may be of any type
    path: optional string containing the complete path of the folder containing the dataset
          default location is the folder of execution of this module.
          Important: string must contain "\\" at the end
          example: "D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\dataset\\"
    name: optional string with the desired name for each file created, the default is matrix
    modifier: an optional object representing a function used to modify the dataset
              the default value is None

    Description:
    Creates the requested quantity of files of type image(png or jpg) or table(csv) representing matrices created
    with the FillMwithSM function at the optional location.
    """

    for i in range(quantity):  # every generated image is a new iteration
        matrix = FillMwithSM(*size)  # creates the matrix using the selected function

        if modifier is None:
            continue
        else:
            matrix = modifier(matrix, mod1, mod2, mod3)

        fileName = path + name + str(i) + "." + fileType  # creates a specific name for each file in the dataset

        if fileType == "csv":
            table = pd.DataFrame(matrix)  # creates a panda dataframe of the matrix
            table.to_csv(fileName, header = None, index = None)  # then exports this dataframe as table and removes index and header

        elif fileType == "jpg" or "png":
            scipy.misc.imsave(fileName, matrix)  # exports matrix as image with the specific name using scipy



"""
Can a Python function be an argument of another function?

Yes.

def myfunc(anotherfunc, extraArgs):
    anotherfunc(*extraArgs)
To be more specific ... with various arguments ...

>>> def x(a,b):
...     print "param 1 %s param 2 %s"%(a,b)
...
>>> def y(z,t):
...     z(*t)
...
>>> y(x,("hello","manuel"))
param 1 hello param 2 manuel
>>>
"""
 # --> use this method to do the last function














"""
def printAll():
    """Short summary.

    Parameters
    ----------
    dataQuantity : type
        Description of parameter `dataQuantity`.

    Returns
    -------
    type
        Description of returned object.

    """
    for i in range(3):
        for j in range(50, 101, 10):
            for k in ["color", "binary", None, "colorBinary"]:
                dataSetGen(quantity = 1,
                           size = (32, 32, 5, 5),
                           fileType = "png",
                           mod1 = j,
                           mod2 = i,
                           mod3 = k,
                           path = "D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\samples\\",
                           name = str(["modifiedLocations = _", i, "___amount = _", j, "___noiseType = _", k]),
                           modifier = addNoise)
"""
"""
def printOne():
    dataSetGen(quantity = 1,
               size = (32, 32, 5, 5),
               fileType = "png",
               mod1 = 54,
               mod2 = 0,
               mod3 = "colorBinary",
               path = "D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\",
               name = "test",
               modifier = addNoise)
"""






















# from matrix gen 3

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
