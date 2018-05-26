from DatasetGenerator import *
# the datasetcreationtool only provides a more intuitive way of creating datasets:
# it is meant to be double-clicked, and asks the user to input the values
# for a dataset. In the end, if all values are valid, the dataset is created.

# this tool adds no functionality to the main functions and classes of the BSP
print("first choose a name to give to the dataset")
filename = str(input("name of the dataset: "))

print("\nchoose a description to give to the dataset")
description = str(input("description of the dataset: "))

print("\nchoose the number of elements that should be in the dataset")
size = int(input("datasetsize: "))

print("\nchoose the Matrix dimensions")
matrixParam = (int(input("amount of columns for the main matrix: ")), int(input("amount of rows for the main matrix: ")))

print("\nchoose the Submatrix dimensions")
subMatrixParam = (int(input("amount of columns for the submatrix: ")), int(input("amount of rows for the submatrix: ")))

print("""\nchoose the amount of noise to be added to the matrix (a value between 0 and 100 is recommended) and where to
      add this noise. (1 to 1's, 0 to 0's, 2 to all parts""")
noiseParam = (int(input("amount of noise to be added to the matrix: ")), int(input("location of noise inside the matrix: ")))

print("------------------------------")
Confirmation = input("do you really want to create the dataset? y/n")
print("------------------------------")


if Confirmation == "y":
    createDataset(size, filename, description, matrixParam, subMatrixParam, noiseParam)
    print("succes")

else:
    print("not created")

keepopen = input()



