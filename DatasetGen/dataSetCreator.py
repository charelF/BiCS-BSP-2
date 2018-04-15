import os
os.chdir("D:\\GOOGLE DRIVE\\School\\sem-2-2018\\BSP2\\BiCS-BSP-2\\DatasetGen")
# somehow the files think they are in another directory than they really are.

from dataSetGenerator import *

Confirmation = input("do you really want to create the dataset? y/n")
if Confirmation == "y":
    createDataset(2000, "test", (20, 20), (5, 5), (25, 0))
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
