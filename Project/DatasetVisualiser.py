import scipy.misc
import DatasetGenerator as dsg
import glob
import numpy as np

for ds in glob.glob("*_input.txt"):

    dataset = dsg.loadDataset(ds)

    delimitterR = np.ones(dataset.shape[1:])
    delimitterG = np.zeros(dataset.shape[1:])
    delimitterB = np.zeros(dataset.shape[1:])
    delimitterRGB = np.stack((delimitterR, delimitterG, delimitterB), axis=2)

    de = delimitterRGB[1:3]
    deV = de.reshape(de.shape[1], de.shape[0], de.shape[2])

    datasetPreview = np.vstack(tuple(dataset[i] for i in range(10)))

    entry0 = np.stack((dataset[0], dataset[0], dataset[0]), axis=2)
    entry1 = np.stack((dataset[1], dataset[1], dataset[1]), axis=2)
    entry2 = np.stack((dataset[2], dataset[2], dataset[2]), axis=2)
    entry3 = np.stack((dataset[3], dataset[3], dataset[3]), axis=2)
    entry4 = np.stack((dataset[4], dataset[4], dataset[4]), axis=2)
    entry5 = np.stack((dataset[5], dataset[5], dataset[5]), axis=2)
    entry6 = np.stack((dataset[6], dataset[6], dataset[6]), axis=2)
    entry7 = np.stack((dataset[7], dataset[7], dataset[7]), axis=2)

    vDelimitterMissingPart = np.stack((np.ones((de.shape[0]*3,de.shape[0])),
                                       np.zeros((de.shape[0]*3,de.shape[0])),
                                       np.zeros((de.shape[0]*3,de.shape[0]))),
                                      axis=2)

    vDelimitter = np.vstack((deV, deV, deV, deV, vDelimitterMissingPart))

    datasetPreviewR = np.vstack((entry1, de, entry3, de, entry5, de, entry7))

    datasetPreviewL = np.vstack((entry0, de, entry2, de, entry4, de, entry6))

    datasetPreviewCombined = np.hstack((datasetPreviewR, vDelimitter, datasetPreviewL))

    scipy.misc.imsave(ds+".png", datasetPreviewCombined)
