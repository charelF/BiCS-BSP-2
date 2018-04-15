import os
os.chdir("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
# somehow the files think they are in another directory than they really are.

from dataSetGenerator import *

q = int(input("datasetsize  "))
a = int(input("column Matrix  "))
b = int(input("row Matrix  "))
c = int(input("column SubMatrix  "))
d = int(input("row SubMatrix  "))
e = int(input("amount of noise  "))
f = int(input("location of noise  "))

Confirmation = input("do you really want to create the dataset? y/n")
if Confirmation == "y":
    createDataset(q, "test", (a, b), (c, d), (e, f))
    print("succes")
    print("input shape is: ", loadDataset("test_input.txt").shape)
    print("output shape is: ", loadDataset("test_output.txt").shape)
    print(" ")
    print(" ")
    print("the first input values are: ", loadDataset("test_input.txt")[0:2])
    print("the first output values are: ", loadDataset("test_output.txt")[0:2])
else:
    print("not created")

keepopen = input()
