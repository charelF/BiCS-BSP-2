import numpy as np   # module used for creating matrices
import random   # module used to find random integers
import pandas as pd   # module used to export matrix as CSV
import scipy.misc   # module used to export matrix as image
import tensorflow as tf

a=np.load("matrix.npy")


print(a.shape, a.size, type(a))
"""
#print(a[345])

#print([i for i in a[234]])


Q=np.arange(100000)
print(Q.size, Q.shape)

table = pd.DataFrame(Q)
table.to_csv("test.csv", header=None, index=None)

#scipy.misc.imsave("test", matrix)

np.save("npytest", Q)"""


#train, test = tf.keras.datasets.mnist.load_data()
#mnist_x, mnist_y = train

mnist_ds = tf.data.Dataset.from_tensor_slices(a)

print(mnist_ds)

