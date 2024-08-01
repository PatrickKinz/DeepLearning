#%%
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, randn, random, uniform
from numpy.random import default_rng,shuffle
rng=default_rng()
from tqdm import tqdm  #for progress bar
import SimpleITK as sitk
import QSM_and_qBOLD_functions as QQ
from tqdm import tqdm  #for progress bar

import nibabel as nib
import os

#%%
# load niftis
subject_name = "RC1155_sub-187"
subject_name_alt = "RC1155_sub187"
path_name1 = "D:/Neurologie/QORMIND/PK/"
suffix_list = ['.nii.gz','_fitted.nii.gz','_fitted_constrained.nii.gz']
suffix=suffix_list[0]
#RC1121_sub-186



names=['S0','R2','Y','DBV','chi_nb','T1']

S0_img = nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF/',names[0],suffix]))
S0 = nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF/',names[0],suffix])).get_fdata()
R2 = nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF/',names[1],suffix])).get_fdata()
Y = nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF/',names[2],suffix])).get_fdata()
DBV = nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF/',names[3],suffix])).get_fdata()
chi_nb = nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF/',names[4],suffix])).get_fdata()
T1 = nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF/',names[5],suffix])).get_fdata()

print(T1.shape)
t=np.array([2.5,6.50,10.50,14.50,18.50,22.50])/1000 #Neurology data
TR = 45.0/1000
alpha = np.array([8,18,30])*2*np.pi/360

def S_T1(S0,T1,alpha,TR):
    epsilon = np.exp(-TR/T1)
    return S0*np.sin(alpha)*(1-epsilon)/(1-np.cos(alpha)*epsilon)


qBOLD8 = QQ.f_qBOLD_GRE_3D(S_T1(S0,T1+0.0000001,alpha[0],TR),R2,Y,DBV,chi_nb,t)
qBOLD18 = QQ.f_qBOLD_GRE_3D(S_T1(S0,T1+0.0000001,alpha[1],TR),R2,Y,DBV,chi_nb,t)


#%%

# load GRE echoes and QSM results
qBOLD_8_list = []
qBOLD_18_list = []
qBOLD_30_list = []

for i in range(6):
    qBOLD_8_list.append(nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/8d/',subject_name_alt,'_ses-mri01_echo-',str(i+1),'_part-mag_8d.nii'])).get_fdata())
    qBOLD_18_list.append(nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/18d/',subject_name_alt,'_ses-mri01_echo-',str(i+1),'_part-mag_18d.nii'])).get_fdata())
    qBOLD_30_list.append(nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/30d/',subject_name_alt,'_ses-mri01_echo-',str(i+1),'_part-mag_30d.nii'])).get_fdata())

qBOLD_8_array = np.array(qBOLD_8_list)
qBOLD_18_array = np.array(qBOLD_18_list)
qBOLD_30_array = np.array(qBOLD_30_list)

max_8d = np.max(qBOLD_8_array)
max_18d = np.max(qBOLD_18_array)
max_30d = np.max(qBOLD_30_array)
max_total = np.max([max_8d,max_18d,max_30d])
#%%
qBOLD8 = qBOLD8*max_total
qBOLD18 = qBOLD18*max_total
#%%
img=nib.Nifti1Image(qBOLD18,S0_img.affine)
nib.save(img,"".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF/qBOLD18',suffix]))                    
# %%
img=nib.Nifti1Image(qBOLD8,S0_img.affine)
nib.save(img,"".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF/qBOLD8',suffix]))  