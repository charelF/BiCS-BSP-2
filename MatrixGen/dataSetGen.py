from matrixGen3 import *

a = FeatureMatrix(8, 9)
print(a.content)
a.fillMatrixWithSubmatrix(3, 4)
print(a.content)
a.addBinaryNoise(33, 0)
print(a.content)
