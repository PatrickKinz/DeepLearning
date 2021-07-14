import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import random_shapes


# These shapes are well suited to test segmentation algorithms. Often, we
# want shapes to overlap to test the algorithm. This is also possible:
image, _ = random_shapes((128, 128), min_shapes=5, max_shapes=10,
                         min_size=20, allow_overlap=True)
image2, _ = random_shapes((128, 128), min_shapes=5, max_shapes=10,
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


Params = np.zeros(image.shape)
Params[:,:,0] = image[:,:,0]*2000.0/255 + image2[:,:,0]*500.0/255
Params[:,:,1] = image[:,:,1]*30.0/255 + image2[:,:,1]*30.0/255
Params[:,:,2] = image[:,:,2]*20.0/255 + image2[:,:,2]*10.0/255


fig, axes = plt.subplots(nrows=1, ncols=3)
ax = axes.ravel()
ax[0].imshow(Params[:,:,0], cmap='Greys')
ax[1].imshow(Params[:,:,1], cmap='Greys')
ax[2].imshow(Params[:,:,2], cmap='Greys')
for a in ax:
    a.set_xticklabels([])
    a.set_yticklabels([])
plt.show()

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


plt.plot(t,Signal[20,25,:],'o')
