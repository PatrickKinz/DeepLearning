# Code to convert BRain Phantom npz to matlab file


#%%
import numpy as np
import scipy.io as io

#%%
data = np.load('D:/Brain_Phantom/Patches_no_air_big_triple_GRE/15GB_1Pnoise_test.npz')                                                                 
list(data.keys())         


#%%

io.savemat('D:/Brain_Phantom/Patches_no_air_big_triple_GRE/15GB_1Pnoise_test.mat', data) 
# %%
