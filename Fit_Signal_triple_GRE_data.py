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
from scipy.optimize import curve_fit

#%%
# load niftis
subject_name = "RC1155_sub-187"
subject_name_alt = "RC1155_sub187"
path_name1 = "D:/Neurologie/QORMIND/PK/"
#RC1121_sub-186


names = ['S0.nii.gz','R2.nii.gz','Y.nii.gz','DBV.nii.gz','chi_nb.nii.gz','T1.nii.gz']

# load qBOLD ANN results
S0_img = nib.load(os.path.join("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF']),names[0]))
S0 = nib.load(os.path.join("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF']),names[0])).get_fdata()
R2 = nib.load(os.path.join("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF']),names[1])).get_fdata()
Y = nib.load(os.path.join("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF']),names[2])).get_fdata()
DBV = nib.load(os.path.join("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF']),names[3])).get_fdata()
chi_nb = nib.load(os.path.join("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF']),names[4])).get_fdata()
T1 = nib.load(os.path.join("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF']),names[5])).get_fdata()

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
qBOLD_8_array = qBOLD_8_array/max_total
qBOLD_18_array = qBOLD_18_array/max_total
qBOLD_30_array = qBOLD_30_array/max_total

QSM_8d = nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/8d/output/QSM_MEDI/',subject_name,'_ses-mri01_8d_QSM_MEDI_Chimap.nii.gz'])).get_fdata()
QSM_18d = nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/18d/output/QSM_MEDI/',subject_name,'_ses-mri01_18d_QSM_MEDI_Chimap.nii.gz'])).get_fdata()
QSM_30d = nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/30d/output/QSM_MEDI/',subject_name,'_ses-mri01_30d_QSM_MEDI_Chimap.nii.gz'])).get_fdata()

mask_8d = nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/8d/output/QSM_MEDI/',subject_name,'_ses-mri01_8d_QSM_MEDI_mask_QSM.nii.gz'])).get_fdata()
mask_18d = nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/18d/output/QSM_MEDI/',subject_name,'_ses-mri01_18d_QSM_MEDI_mask_QSM.nii.gz'])).get_fdata()
mask_30d = nib.load("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/30d/output/QSM_MEDI/',subject_name,'_ses-mri01_30d_QSM_MEDI_mask_QSM.nii.gz'])).get_fdata()

mask_combined = mask_8d * mask_18d * mask_30d 

#%%

print(T1.shape)
t=np.array([2.5,6.50,10.50,14.50,18.50,22.50])/1000 #Neurology data
TR = 45.0/1000
alpha = np.array([8,18,30])*2*np.pi/360

def S_T1(S0,T1,alpha,TR):
    epsilon = np.exp(-TR/T1)
    return S0*np.sin(alpha)*(1-epsilon)/(1-np.cos(alpha)*epsilon)


def qqBOLD_triple_GRE(t,S0,R2,Y,DBV,chi_nb,T1):
    t=np.array([2.5,6.50,10.50,14.50,18.50,22.50])/1000 #Neurology data
    TR = 45.0/1000
    alpha = np.array([8,18,30])*2*np.pi/360
    output = []
    output.extend( QQ.f_qBOLD_GRE_1value(S_T1(S0,T1+0.0000001,alpha[0],TR),R2,Y,DBV,chi_nb,t))
    output.extend( QQ.f_qBOLD_GRE_1value(S_T1(S0,T1+0.0000001,alpha[1],TR),R2,Y,DBV,chi_nb,t))
    output.extend( QQ.f_qBOLD_GRE_1value(S_T1(S0,T1+0.0000001,alpha[2],TR),R2,Y,DBV,chi_nb,t))
    QSM = QQ.f_QSM(Y,DBV,chi_nb)
    output.extend(QSM)
    output.extend(QSM)
    output.extend(QSM)
    
    return np.array(output)

#qBOLD18 = QQ.f_qBOLD_GRE_3D(S_T1(S0,T1+0.0000001,alpha[1],TR),R2,Y,DBV,chi_nb,t)

#%% curve fit test
x = 200
y = 320
z = 40
x_data = t
y_data = np.concatenate((qBOLD_8_array,qBOLD_18_array,qBOLD_30_array,np.expand_dims(QSM_8d,axis=0),np.expand_dims(QSM_18d,axis=0),np.expand_dims(QSM_30d,axis=0))) 
#y_data = np.array([qBOLD_8_array[:,x,y,z],qBOLD_18_array[:,x,y,z],qBOLD_30_array[:,x,y,z],QSM_8d[x,y,z],QSM_18d[x,y,z],QSM_30d[x,y,z],])

plt.plot(y_data[:,x,y,z],'.')

#%% test if function works
y_data_calculated = qqBOLD_triple_GRE(t,S0[x,y,z],R2[x,y,z],Y[x,y,z],DBV[x,y,z],chi_nb[x,y,z],T1[x,y,z])

plt.plot(y_data_calculated,'.')
#%% fit one voxel as test
p0=np.array([S0[x,y,z],R2[x,y,z],Y[x,y,z],DBV[x,y,z],chi_nb[x,y,z],T1[x,y,z]])
lower_bounds=np.array([0.1,1,0,0,-0.3,0.1])
upper_bounds=np.array([1.5,40,1,0.1,0.3,3])
popt,pcov = curve_fit(qqBOLD_triple_GRE,x_data,y_data[:,x,y,z],p0,bounds=(lower_bounds,upper_bounds))
y_data_calculated_fit = qqBOLD_triple_GRE(t,*popt)


lower_bounds2=p0*0.9
upper_bounds2=p0*1.1
if p0[4]<0:
    lower_bounds2[4] = p0[4]*1.1
    upper_bounds2[4] = p0[4]*0.9
popt2,pcov2 = curve_fit(qqBOLD_triple_GRE,x_data,y_data[:,x,y,z],p0,bounds=(lower_bounds2,upper_bounds2))
y_data_calculated_fit2 = qqBOLD_triple_GRE(t,*popt2)

plt.figure()
plt.plot(y_data[:,x,y,z],'b.')
plt.plot(y_data_calculated,'r.')
plt.plot(y_data_calculated_fit,'g.')
plt.plot(y_data_calculated_fit2,'k.')

#%% physical bounds and relative bounds
                                #S0 , R2, Y,DBV,chi_nb,T1
physical_lower_bounds = np.array([0.1,  0, 0,  0,  -0.3,0.1])
physical_upper_bounds= np.array([1.5, 50, 1,0.1,   0.3,10])
minimal_space        = np.array([0.1, 1, 0.1,0.04, 0.05,0.1])

def check_start_params_for_physical_bounds(p0,lower_b,upper_b,min_sp):
    for i in range(len(p0)):
        if p0[i] < lower_b[i]:
            p0[i] = lower_b[i]+min_sp[i]
        if p0[i] > upper_b[i]:
            p0[i] = upper_b[i]-min_sp[i]
    return p0

def constrain_close_to_p0(p0,phys_lower_b,phys_upper_b,min_sp):
    lower_bounds=p0*0.9
    upper_bounds=p0*1.1
    if p0[4]<0:
        lower_bounds[4] = p0[4]*1.1
        upper_bounds[4] = p0[4]*0.9
    for i in range(len(p0)):
        if lower_bounds[i] > (p0[i]-min_sp[i]):
            lower_bounds[i] = p0[i]-min_sp[i]
        if lower_bounds[i] < phys_lower_b[i]:
            lower_bounds[i] = phys_lower_b[i]
        if upper_bounds[i] < (p0[i]+min_sp[i]):
            upper_bounds[i] = p0[i]+min_sp[i]
        if upper_bounds[i] > phys_upper_b[i]:
            upper_bounds[i] = phys_upper_b[i]
    return (lower_bounds,upper_bounds)
        
        

#%% fit whole brain
popt_array = np.zeros((6,QSM_8d.shape[0],QSM_8d.shape[1],QSM_8d.shape[2]))
popt2_array = np.zeros((6,QSM_8d.shape[0],QSM_8d.shape[1],QSM_8d.shape[2]))
for x in tqdm(range(QSM_8d.shape[0])):
    for y in range(QSM_8d.shape[1]):
        for z in range(QSM_8d.shape[2]):
            if mask_combined[x,y,z]>0:
                p0=np.array([S0[x,y,z],R2[x,y,z],Y[x,y,z],DBV[x,y,z],chi_nb[x,y,z],T1[x,y,z]])
                p0 = check_start_params_for_physical_bounds(p0,physical_lower_bounds,physical_upper_bounds,minimal_space)
                tight_bounds = constrain_close_to_p0(p0,physical_lower_bounds,physical_upper_bounds,minimal_space)
                try:
                    popt,pcov = curve_fit(qqBOLD_triple_GRE,x_data,y_data[:,x,y,z],p0,bounds=(physical_lower_bounds,physical_upper_bounds))
                    popt_array[:,x,y,z]=popt
                except:
                    pass
                try:
                    popt2,pcov2 = curve_fit(qqBOLD_triple_GRE,x_data,y_data[:,x,y,z],p0,bounds=tight_bounds)
                    popt2_array[:,x,y,z]=popt2
                except:
                    pass
        

#%%
names=['S0','R2','Y','DBV','chi_nb','T1']
for i in range(len(names)):
    img=nib.Nifti1Image(popt_array[i,:,:,:],S0_img.affine)
    nib.save(img,"".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF/',names[i],'_fitted.nii.gz']))
for i in range(len(names)):
    img=nib.Nifti1Image(popt2_array[i,:,:,:],S0_img.affine)
    nib.save(img,"".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF/',names[i],'_fitted_constrained.nii.gz']))                    
# %%
#code finished running after 31h:12min