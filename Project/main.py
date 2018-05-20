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

# print("\n\n\n========================= DataSetInfo: =========================\n")
# print("Size:                ", inputData.shape)
# print("Type:                ", type(inputData))
# print("InputName:           ", glob.glob("*_input.txt")[0])
# print("InputDescription:    ", dsg.loadDatasetDescription(glob.glob("*_input.txt")[0]))
# print("OutputName:          ", glob.glob("*_output.txt")[0])
# print("OutputDescription:   ", dsg.loadDatasetDescription(glob.glob("*_output.txt")[0]))
# print("\n================================================================\n\n\n")

# print("\n\n\n========================= DataSetInfo: =========================\n")
# print("Size:                ", inputData.shape)
# print("Type:                ", type(inputData))
# print("InputName:           ", glob.glob("*_input.txt")[0])
# print("InputDescription:    ", dsg.loadDatasetDescription(glob.glob("*_input.txt")[0]))
# print("OutputName:          ", glob.glob("*_output.txt")[0])
# print("OutputDescription:   ", dsg.loadDatasetDescription(glob.glob("*_output.txt")[0]))
# print("\n================================================================\n\n\n")
