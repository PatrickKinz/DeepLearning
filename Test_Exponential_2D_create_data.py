import matplotlib.pyplot as plt
import numpy as np
import h5py
from skimage.draw import random_shapes
from scipy.ndimage import gaussian_filter
from numpy.random import rand, randn
from tqdm import tqdm
"""
Function to test creating 2D data of Signal with S0, T2 and T2S
"""
Image_size=[128,128]

# %%
# These shapes are well suited to test segmentation algorithms. Often, we
# want shapes to overlap to test the algorithm. This is also possible:
image, _ = random_shapes(Image_size, min_shapes=5, max_shapes=10,
                         min_size=20, allow_overlap=True)

# We can visualize the images.
fig, axes = plt.subplots(nrows=2, ncols=2)
ax = axes.ravel()
ax[0].imshow(image)
ax[1].imshow(image[:,:,0], cmap='Reds')
ax[2].imshow(image[:,:,1], cmap='Greens')
ax[3].imshow(image[:,:,2], cmap='Blues')
for a in ax:
    a.set_xticklabels([])
    a.set_yticklabels([])
plt.show()
# %%

image2, _ = random_shapes(Image_size, min_shapes=5, max_shapes=10,
                         min_size=20, allow_overlap=True)
# We can visualize the images.
fig, axes = plt.subplots(nrows=2, ncols=2)
ax = axes.ravel()
ax[0].imshow(image2)
ax[1].imshow(image2[:,:,0], cmap='Reds')
ax[2].imshow(image2[:,:,1], cmap='Greens')
ax[3].imshow(image2[:,:,2], cmap='Blues')
for a in ax:
    a.set_xticklabels([])
    a.set_yticklabels([])
plt.show()

# %% calculate S0, T2, T2S from images

Params = np.zeros(image.shape)
Params[:,:,0] = image[:,:,0]*2000.0/255 + image2[:,:,0]*500.0/255 #S0
Params[:,:,1] = image[:,:,1]*20.0/255 + image2[:,:,1]*20.0/255 #T2
Params[:,:,2] = image[:,:,2]*10.0/255 + image2[:,:,2]*10.0/255 #T2S
Params[:,:,1] = Params[:,:,1] + Params[:,:,2] #ensure T2 is larger than T2S

fig, axes = plt.subplots(nrows=2, ncols=3)
ax = axes.ravel()
ax[0].imshow(Params[:,:,0], cmap='Greys')
ax[0].title.set_text('S0')
ax[1].imshow(Params[:,:,1], cmap='Greys')
ax[1].title.set_text('T2')
ax[2].imshow(Params[:,:,2], cmap='Greys')
ax[2].title.set_text('T2S')
ax[3].hist(Params[:,:,0].ravel())
ax[4].hist(Params[:,:,1].ravel())
ax[5].hist(Params[:,:,2].ravel())
plt.show()

# %%
Params = gaussian_filter(Params, sigma = ([2,2,0])) # blur 2D images but not from layer to layer

fig, axes = plt.subplots(nrows=2, ncols=3)
ax = axes.ravel()
ax[0].imshow(Params[:,:,0], cmap='Greys')
ax[0].title.set_text('S0')
ax[1].imshow(Params[:,:,1], cmap='Greys')
ax[1].title.set_text('T2')
ax[2].imshow(Params[:,:,2], cmap='Greys')
ax[2].title.set_text('T2S')
ax[3].hist(Params[:,:,0].ravel())
ax[4].hist(Params[:,:,1].ravel())
ax[5].hist(Params[:,:,2].ravel())
plt.show()

# %% calculate Signal

def simulateSignal(S0,T2,T2S,t):
    output = np.zeros(len(t))
    for i in range(6):
        output[i] = S0 * np.exp(-t[i]/T2S)
    for i in range(6,len(t)):
        output[i] = S0 * np.exp(-abs((40-t[i]))*(1/T2S - 1/T2) - t[i]/(T2) )
    return output

t=np.array([3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48])
Signal = np.zeros((128,128,16))
for i in range(128):
    for j in range(128):
        Signal[i,j,:] = simulateSignal(Params[i,j,0],Params[i,j,1],Params[i,j,2],t)

fig, axes = plt.subplots(nrows=4, ncols=4)
ax = axes.ravel()
for i in range(16):
    ax[i].imshow(Signal[:,:,i], cmap='Greys')
for a in ax:
    a.set_xticklabels([])
    a.set_yticklabels([])
plt.show()

# %% Plot 5 random choosen voxels
for i in range(5):
    plt.plot(t,Signal[np.random.randint(0,Image_size[0]),np.random.randint(0,Image_size[1]),:],'o')

# %% create giant array and save it in hdf5
def addNoise(input,Spread,Offset):
    output = np.multiply(input, 1 + Spread*randn(input.size).reshape(input.shape)) + Offset*rand(input.size).reshape(input.shape)
    return output


def SimulateLayer(Image_size,t):
    Params= np.zeros((Image_size[0],Image_size[1],3))
    image, _ = random_shapes(Image_size, min_shapes=5, max_shapes=10,
                             min_size=20, allow_overlap=True)
    image2, _ = random_shapes(Image_size, min_shapes=5, max_shapes=10,
                             min_size=20, allow_overlap=True)
    Params[:,:,0] = image[:,:,0]*2000.0/255 + image2[:,:,0]*500.0/255 #S0
    Params[:,:,1] = image[:,:,1]*20.0/255 + image2[:,:,1]*20.0/255 #T2
    Params[:,:,2] = image[:,:,2]*10.0/255 + image2[:,:,2]*10.0/255 #T2S
    Params[:,:,1] = Params[:,:,1] + Params[:,:,2] #ensure T2 is larger than T2S
    Params = gaussian_filter(Params, sigma = ([2,2,0])) # blur 2D images but not from layer to layer

    Signal= np.zeros((Image_size[0],Image_size[1],len(t)))
    Signal_noise= np.zeros((Image_size[0],Image_size[1],len(t)))
    for i in range(Image_size[0]):
        for j in range(Image_size[1]):
            Signal[i,j,:] = simulateSignal(Params[i,j,0],Params[i,j,1],Params[i,j,2],t)
        Signal_noise[i,:,:] = addNoise(Signal[i,:,:], 0.05, 100)
    return  Params, Signal, Signal_noise

a, b, c = SimulateLayer(Image_size,t)

plt.imshow(a[:,:,2], cmap='Greys')
# %%
plt.figure()
plt.plot(b[1,1,:],'o-')
plt.plot(c[1,1,:],'o')
plt.show()
# %%
N_train = 1000
N_test = 100
with h5py.File("Exponential2D_bigger.hdf5", "a") as f:
    dset_input_train = f.create_dataset("input_train", (N_train,Image_size[0],Image_size[1],len(t)), dtype='f')
    dset_input_noise_train = f.create_dataset("input_noise_train", (N_train,Image_size[0],Image_size[1],len(t)), dtype='f')
    dset_target_train = f.create_dataset("target_train", (N_train,Image_size[0],Image_size[1],3), dtype='f')
    dset_input_test = f.create_dataset("input_test", (N_test,Image_size[0],Image_size[1],len(t)), dtype='f')
    dset_input_noise_test = f.create_dataset("input_noise_test", (N_test,Image_size[0],Image_size[1],len(t)), dtype='f')
    dset_target_test = f.create_dataset("target_test", (N_test,Image_size[0],Image_size[1],3), dtype='f')

    for i in tqdm(range(N_train)):
        dset_target_train[i,:,:,:],dset_input_train[i,:,:,:],dset_input_noise_train[i,:,:,:]=SimulateLayer(Image_size,t)
    for i in tqdm(range(N_test)):
        dset_target_test[i,:,:,:],dset_input_test[i,:,:,:],dset_input_noise_test[i,:,:,:]=SimulateLayer(Image_size,t)



#%% Test Read the file
f = h5py.File('Exponential2D.hdf5', 'r')
list(f.keys())
dset = f['input_noise_test']
dset.shape
plt.plot(dset[0,1,1,:],'o')

f.close()
