import os
os.chdir("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
# somehow the files think they are in another directory than they really are.

from dataSetGenerator import *
h = str(input("name of the dataset  "))
g = str(input("description of the dataset, used to verfiy we are training on the right one  "))
q = int(input("datasetsize  "))
a = int(input("column Matrix  "))
b = int(input("row Matrix  "))
c = int(input("column SubMatrix  "))
d = int(input("row SubMatrix  "))
e = int(input("amount of noise  "))
f = int(input("location of noise  "))

Confirmation = input("do you really want to create the dataset? y/n")
if Confirmation == "y":
    createDataset(q, h, (a, b), (c, d), (e, f), (g+"name: ", h, "noise and location: ", str(e), str(f)))
    print("succes")
    print("input shape is: ", loadDataset("{}_input.txt".format(h)).shape)
    print("output shape is: ", loadDataset("{}_output.txt".format(h)).shape)
    print("description of the dataset is: ", loadDatasetDescription("{}_input.txt".format(h)))
    print("description of the dataset is: ", loadDatasetDescription("{}_output.txt".format(h)))
    print(" ")
    print(" ")
    print("the first input values are: ", loadDataset("{}_input.txt".format(h))[0:2])
    print("the first output values are: ", loadDataset("{}_output.txt".format(h))[0:2])
else:
    print("not created")

keepopen = input()
