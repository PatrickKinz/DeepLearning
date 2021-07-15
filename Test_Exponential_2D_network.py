from tensorflow.keras.datasets import cifar10

# Load CIFAR10 dataset
(input_train, target_train), (input_test, target_test) = cifar10.load_data()




# %%

input_train.shape
input_train.dtype


# %%

import h5py
import numpy as np
with h5py.File("mytestfile.hdf5", "w") as f:
    dset = f.create_dataset("mydataset", (100,), dtype='i')


# %%
f.close()
f = h5py.File('mytestfile.hdf5', 'a')
list(f.keys())
dset = f['mydataset']
dset.shape
dset.dtype
dset[10]
dset[...] = np.arange(100)
