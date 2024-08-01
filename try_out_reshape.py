import numpy as np

a=np.zeros((2,3,3,16))
a
for i in range(16):
    a[:,:,:,i]=i
a
b=a.reshape(-1,16)
b
b.shape
