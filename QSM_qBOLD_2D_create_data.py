import matplotlib.pyplot as plt
import numpy as np
import h5py
from skimage.draw import random_shapes
from scipy.ndimage import gaussian_filter
from numpy.random import rand, randn, random

from tqdm import tqdm
import SimpleITK as sitk
import QSM_and_qBOLD_functions as QQ
#%%
"""
Function to test creating 2D data of Signal with S0, T2 and T2S
"""

#seg = sitk.ReadImage("C:/Users/pk24/Documents/Programming/Brain_Phantom/Segmentation.TIF")
seg = sitk.ReadImage("../Brain_Phantom/Segmentation.TIF")
print(seg.GetSize())
print(seg.GetOrigin())
print(seg.GetSpacing())
#Spacing is wrong: Paper says segmentation is (1,1,1). Based on data thats (0.9375,0.9375,3)
print(seg.GetDirection())
print(seg.GetNumberOfComponentsPerPixel())
print(seg.GetWidth())
print(seg.GetHeight())
print(seg.GetDepth())
print(seg)
print(seg.GetSize()[0]*seg.GetSpacing()[0])
seg.GetPixel(255,255,147)
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
nda_seg = sitk.GetArrayViewFromImage(seg)
nda_seg.shape
#%%
plt.imshow(nda_seg[70,:,:])
plt.colorbar()
#%%
plt.imshow(nda_seg[70,100:150,100:150])
plt.colorbar()
#%%
""" Take Parameters and Signal fill 3D brain, repeat N times """
M = 1
for m in range(M):
    """ create parameter values for each tissue type """
    #currently assuming same parameter distribution for all tissue types
    N_tissues=17 #Air and Abnormal_WM not included
    t=np.array([3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48])/1000

    a=0.5*np.ones(N_tissues) #S0
    b=random(N_tissues) #R2
    c=random(N_tissues) #Y
    d=random(N_tissues) #nu
    e=random(N_tissues) #chi_nb
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

    """ Save Parameters as truth/targets and signal as input/targets for CNN """
    #Save one brain with 5 Parameters(a,b,c,d,e) in each voxels
    brainParams = sitk.Image(seg.GetSize(),sitk.sitkVectorFloat32,5)
    #one brain with Signal in each voxel
    brainSignal = sitk.Image(seg.GetSize(),sitk.sitkVectorFloat32,16)
    #one brain with QSM in each voxel
    brainQSM = sitk.Image(seg.GetSize(),sitk.sitkVectorFloat32,1)
    #loop over brain.
    for i in range(seg.GetWidth()):
        for j in range(seg.GetHeight()):
            for k in range(seg.GetDepth()):
                type = seg.GetPixel(i,j,k)
                if type == 0:
                    continue #0 is empty/air
                else:
                    type=type-1 #shift to leave out 0
                    brainParams.SetPixel(i,j,k,[a[type],b[type],c[type],d[type],e[type]])
                    brainSignal.SetPixel(i,j,k,signal[type])
                    brainQSM.SetPixel(i,j,k,QSM[type])
    sitk.WriteImage(brainParams,"C:/Users/pk24/Documents/Programming/Brain_Phantom/BrainParams_test_comp.TIF" ,useCompression=True)
    sitk.WriteImage(brainSignal,"C:/Users/pk24/Documents/Programming/Brain_Phantom/BrainSignal_test_comp.TIF" ,useCompression=True)
    sitk.WriteImage(brainQSM,"C:/Users/pk24/Documents/Programming/Brain_Phantom/BrainQSM_test_comp.TIF" ,useCompression=True)
#%%
nda_test = sitk.GetArrayViewFromImage(brainSignal)
nda_test.shape
plt.imshow(nda_test[70,:,30:230,5])
plt.colorbar()
fig=plt.figure()
for i in range(N_tissues):
    plt.plot(t*1000,signal[i,:],'.--',label=i)
fig.legend(loc=7)
fig=plt.figure()
for i in range(N_tissues):
    plt.plot(t*1000,signal[i,:]-S0[i]*np.exp(-R2[i]*t),'.--',label=i)
fig.legend(loc=7)
Y
nu

#%%
def check_if_sorting_by_ParamX_is_relevant_for_signal_shape(Param_to_sort): #a/S0, b/R2, c/Y, d/nu, e/chi_nb, QSM
    # Look at unsorted signal
    plt.figure()
    plt.imshow(signal)
    plt.colorbar()
    # Print Param to sort by
    print(Param_to_sort)

    Anzahl_vector=-np.ones(len(Param_to_sort))
    for i in range(len(Param_to_sort)):
        Anzahl=0
        for j in range(len(Param_to_sort)):
            if i != j:
                if Param_to_sort[i] > Param_to_sort[j]:
                    Anzahl=Anzahl+1
        Anzahl_vector[i]=Anzahl
        #QSM_sorted_QSM[Anzahl]=QSM[i]
        #signal_sorted_by_QSM[Anzahl,:] = signal[i,:]

    #print(Anzahl_vector)

    Param_sorted=np.zeros(len(Param_to_sort))
    signal_sorted_by_Param=np.zeros(signal.shape)
    for i in range(len(Param_to_sort)):
            Param_sorted[int(Anzahl_vector[i])] = Param_to_sort[i]
            signal_sorted_by_Param[int(Anzahl_vector[i]),:] = signal[i,:]-S0[i]*np.exp(-R2[i]*t)
    print(Param_sorted)
    # #Sorting the signals by their QSM value does not show any clear correlation
    plt.figure()
    plt.imshow(signal_sorted_by_Param)
    plt.colorbar()

check_if_sorting_by_ParamX_is_relevant_for_signal_shape(c)

#%%
""" Pull M random slices from 3D matrix and save them (total number M*N)"""
""" Augment data total number ?*M*N """
#Rotation in each Ebene
#Varied cuts. Cor, Sag, Trans or even in between?
#Varied size?
#Deformation?



#%%
""" Loop over saved slices """
    """ Add noise to signal slices, repeat K times (total number K*?*M*N)"""

    """ Norm signal slices """

    """ Save noisy normed slices """
