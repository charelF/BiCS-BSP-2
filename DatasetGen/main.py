import os
os.chdir("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
# somehow the files think they are in another directory than they really are.

# easy way to create a dataset
from dataSetGenerator import *
filename = str(input("name of the dataset  "))
description = str(input("description of the dataset, used to verfiy we are training on the right one  "))
size = int(input("datasetsize  "))
matrixParam = (int(input("column Matrix  ")), int(input("row Matrix  ")))
subMatrixParam = (int(input("column SubMatrix  ")), int(input("row SubMatrix  ")))
noiseParam = (int(input("amount of noise  ")), int(input("location of noise  ")))

Confirmation = input("do you really want to create the dataset? y/n")

if Confirmation == "y":
    createDataset(size, filename, description, matrixParam, subMatrixParam, noiseParam)
    print("succes")

else:
    print("not created")

keepopen = input()




# """description = g + "parameters: " + "datasetsize = " + str(q) +
#               ", dim Matrix = " + str(a) + " ," + str(b) +
#               ", dim SubMatrix = " + str(c) + " ," + str(d) +
#               ", noiseparam = " + str(e) + " ," + str(f))"""
# print("input shape is: ", loadDataset("{}_input.txt".format(h)).shape)
# print("output shape is: ", loadDataset("{}_output.txt".format(h)).shape)
# print("description of the dataset is: ", loadDatasetDescription("{}_input.txt".format(h)))
# print("description of the dataset is: ", loadDatasetDescription("{}_output.txt".format(h)))
# print(" ")
# print(" ")
# print("the first input values are: ", loadDataset("{}_input.txt".format(h))[0:2])
# print("the first output values are: ", loadDataset("{}_output.txt".format(h))[0:2])
