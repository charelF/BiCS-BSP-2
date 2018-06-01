import scipy.misc
import DatasetGenerator as dsg
import glob
import numpy as np

# this code should not be reviewd as it has been created in a very limited
# amount of time due do time contraints at the end of the BSP. It is included
# in the project to given an easy way of visualising newly created datasets

# to execute it, it is only necessary that there is a datasetgenerator.loadadataset
# recognizable dataset combination in the project folder, then just execute the script

# datasets need to have at least 8 entries and at least of size 2 by 2

for ds in glob.glob("*_input.txt"):

    dataset = dsg.loadDataset(ds)

    # creating a red colored delimitter matrix, used to display ds boundaries
    delimitterR = np.ones(dataset.shape[1:])
    delimitterG = np.zeros(dataset.shape[1:])
    delimitterB = np.zeros(dataset.shape[1:])
    delimitterRGB = np.stack((delimitterR, delimitterG, delimitterB), axis=2)

    deHor = delimitterRGB[1:3]  # HORizontal DElinmitter
    deVer = deHor.reshape(deHor.shape[1], deHor.shape[0], deHor.shape[2])
    # VERtical DElinmitter

    entry0 = np.stack((dataset[0], dataset[0], dataset[0]), axis=2)
    entry1 = np.stack((dataset[1], dataset[1], dataset[1]), axis=2)
    entry2 = np.stack((dataset[2], dataset[2], dataset[2]), axis=2)
    entry3 = np.stack((dataset[3], dataset[3], dataset[3]), axis=2)
    entry4 = np.stack((dataset[4], dataset[4], dataset[4]), axis=2)
    entry5 = np.stack((dataset[5], dataset[5], dataset[5]), axis=2)
    entry6 = np.stack((dataset[6], dataset[6], dataset[6]), axis=2)
    entry7 = np.stack((dataset[7], dataset[7], dataset[7]), axis=2)

    deVerMissingPart = np.stack((np.ones((deHor.shape[0]*3,deHor.shape[0])),
                                          np.zeros((deHor.shape[0]*3,deHor.shape[0])),
                                          np.zeros((deHor.shape[0]*3,deHor.shape[0]))),
                                         axis=2)

    deVerCombined = np.vstack((deVer, deVer, deVer, deVer, deVerMissingPart))

    datasetPreviewR = np.vstack((entry1, deHor, entry3, deHor, entry5, deHor, entry7))

    datasetPreviewL = np.vstack((entry0, deHor, entry2, deHor, entry4, deHor, entry6))

    datasetPreviewCombined = np.hstack((datasetPreviewR, deVerCombined, datasetPreviewL))

    scipy.misc.imsave(ds[:-4]+".png", datasetPreviewCombined)
