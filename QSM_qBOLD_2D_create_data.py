import matplotlib.pyplot as plt
import numpy as np
import h5py
from skimage.draw import random_shapes
from scipy.ndimage import gaussian_filter
from numpy.random import rand, randn, random
from tqdm import tqdm
import SimpleITK as sitk
import QSM_and_qBOLD_functions as QQ
"""
Function to test creating 2D data of Signal with S0, T2 and T2S
"""

#seg = sitk.ReadImage("C:/Users/pk24/Documents/Programming/Brain_Phantom/Segmentation.TIF")
seg = sitk.ReadImage("../Brain_Phantom/Segmentation.TIF")
nda_seg = sitk.GetArrayViewFromImage(seg)
"""
MODEL_NV_1_148_SLICES
Tissue_Name                 ,Volume_(cc)    ,Color_code_(RGB) ,Label
Nothing/Air                 ,??             ,(  0,  0,  0)    , 0
GM                          ,854.86         ,(127,127,127)    , 1
WM                          ,572.62         ,(255,255,255)    , 2
CSF                         ,182.45         ,(000,000,180)    , 3
Putamen                     ,9.73           ,(000,120,000)    , 4
Pallidus                    ,3.99           ,(255,000,000)    , 5
Thalamus                    ,13.95          ,(080,120,080)    , 6
Caudatus                    ,10.35          ,(000,110,000)    , 7
Nigra                       ,1.19           ,(200,000,000)    , 8
Red_Nucleus                 ,0.66           ,(255,050,000)    , 9
Dentate_Nucleus             ,1.67           ,(128,255,000)    ,10
Low_PD                      ,825.68         ,(190,000,255)    ,11
Fat                         ,409.18         ,(255,190,190)    ,12
Muscle                      ,552.09         ,(128,000,000)    ,13
Vitreous_Humor              ,14.25          ,(000,000,255)    ,14
Extra_Cranial_Connective    ,54.20          ,(127,127,255)    ,15
Extra_Cranial_Fluid         ,16.54          ,(000,180,180)    ,16
Intra_Cranial_Connective    ,20.04          ,(128,080,255)    ,17
Abnormal_WM                 ,0.00           ,(255,255,000)    ,18
"""

#%%
""" Pull N random partial slices from 3D matrix and save them """
#Rotation in each Ebene
#Varied cuts. Cor, Sag, Trans or even in between?
#Varied size?
#Deformation?
#%%
""" Take Parameters and Signal fill 2D slices, repeat M times for each Slice (total number M*N)"""
for m in range(M):
    """ create parameter values for each tissue type """
    N_tissues=17 #Air and Abnormal_WM not included
    t=np.array([3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48])/1000

    b=random(N_tissues)
    a=0.5*np.ones(b.shape)
    c=random(N_tissues)
    d=random(N_tissues)
    e=random(N_tissues)

    """ Calculate Signal for each tissue type """
    S0 = a   #S0     = 1000 + 200 * randn(N).T
    R2 = (30-1) * b + 1
    SaO2 = 0.98
    Y  = (SaO2 - 0.01) * c + 0.01
    nu = (0.1 - 0.001) * d + 0.001
    chi_nb = ( 0.1-(-0.1) ) * e - 0.1

    signal = QQ.f_qBOLD(S0,R2,Y,nu,chi_nb,t)
    print('signal', signal.shape)
    QSM = QQ.f_QSM(Y,nu,chi_nb)
    print('QSM',QSM.shape)

    """ Save Parameters as truth/targets and signal as input """





#%%
""" Loop over saved slices """
    """ Add noise to signal slices, repeat K times (total number K*M*N)"""

    """ Norm signal slices """

    """ Save noisy normed slices """
