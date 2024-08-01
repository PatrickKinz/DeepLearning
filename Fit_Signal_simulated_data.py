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
import QSM_qBOLD_2D_plotting_functions as QQplt

#%%
Dataset_test=np.load("D:/Brain_Phantom/Patches_no_air_big_GESSE/15GB_0Pnoise_test.npz")
a=Dataset_test['S0']
a.shape
b=Dataset_test['R2']
c=Dataset_test['Y']
d=Dataset_test['nu']
e=Dataset_test['chi_nb']
qBOLD_test=Dataset_test['qBOLD']
QSM_test=Dataset_test['QSM']

S0 = a   #S0     = 1000 + 200 * randn(N).T
R2 = (30-1) * b + 1
SaO2 = 0.98
Y  = (SaO2 - 0.01) * c + 0.01
DBV = (0.1 - 0.001) * d + 0.001
chi_nb = ( 0.1-(-0.1) ) * e - 0.1

print(qBOLD_test.shape)

#%%


t=np.array([29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91])/1000
    
def f_qBOLD_basic(t,S0,R2s ):
    return S0*np.exp(-t*R2s)

def qqBOLD_GESSE(t,S0,R2,Y,DBV,chi_nb):
    t=np.array([29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91])/1000
    output = []
    QBOLD =  QQ.f_qBOLD_GESSE_1value(S0,R2,Y,DBV,chi_nb,t)
    QBOLD = QBOLD.squeeze()
    #print(QBOLD.shape)
    output.extend(QBOLD)
    
    QSM = QQ.f_QSM(Y,DBV,chi_nb)
    #print(QSM.shape)
    output.extend(QSM)
    
    return np.array(output)

#qBOLD18 = QQ.f_qBOLD_GRE_3D(S_T1(S0,T1+0.0000001,alpha[1],TR),R2,Y,DBV,chi_nb,t)

#%% curve fit test
x = 1
y = 1
z = 1
x_data = t
y_data = np.concatenate((qBOLD_test,QSM_test),axis=-1) 

plt.plot(y_data[x,y,z,:],'.')

#%% test if function works
y_data_calculated = qqBOLD_GESSE(t,*S0[x,y,z],*R2[x,y,z],*Y[x,y,z],*DBV[x,y,z],*chi_nb[x,y,z])

plt.plot(y_data_calculated,'.')
#%% fit one voxel as test
p0=np.array([S0[x,y,z],R2[x,y,z],Y[x,y,z],DBV[x,y,z],chi_nb[x,y,z]])
p0=np.squeeze(p0)

lower_bounds=np.array([0.1, 1,0, 0,-0.3])
upper_bounds=np.array([1.5,40,1,0.2,0.3])
popt,pcov = curve_fit(qqBOLD_GESSE,x_data,y_data[x,y,z,:],p0,bounds=(lower_bounds,upper_bounds))
y_data_calculated_fit = qqBOLD_GESSE(t,*popt)


lower_bounds2=p0*0.9
upper_bounds2=p0*1.1
if p0[4]<0:
    lower_bounds2[4] = p0[4]*1.1
    upper_bounds2[4] = p0[4]*0.9
popt2,pcov2 = curve_fit(qqBOLD_GESSE,x_data,y_data[x,y,z,:],p0,bounds=(lower_bounds2,upper_bounds2))
y_data_calculated_fit2 = qqBOLD_GESSE(t,*popt2)

popt_basic,pcov_basic = curve_fit(f_qBOLD_basic,x_data,y_data[x,y,z,:-1],p0=[0.5,15],bounds=([0.1,1],[1.5,30]))
print(popt_basic)
p1 = np.array([popt_basic[0],popt_basic[1],0.5,0.05,0])
popt3,pcov3 = curve_fit(qqBOLD_GESSE,x_data,y_data[x,y,z,:],p1,bounds=(lower_bounds,upper_bounds))
y_data_calculated_fit3 = qqBOLD_GESSE(t,*popt3)

plt.figure()
plt.plot(y_data[x,y,z,:],'b.')
plt.plot(y_data_calculated,'r.')
plt.plot(y_data_calculated_fit,'g.')
plt.plot(y_data_calculated_fit2,'k.')
plt.plot(y_data_calculated_fit3,'m.')



#%% physical bounds and relative bounds
                                #S0 , R2, Y,DBV,chi_nb
physical_lower_bounds = np.array([0.1,  0, 0,  0,  -0.3])
physical_upper_bounds= np.array([1.5, 50, 1,0.1,   0.3])
minimal_space        = np.array([0.1, 1, 0.1,0.04, 0.05])

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
        
        

#%% fit whole brain voxel by voxel
n_slices=5000 #qBOLD_test.shape[0]
offset = 0
popt_array = np.zeros((5,n_slices,qBOLD_test.shape[1],qBOLD_test.shape[2]))
popt2_array = np.zeros((5,n_slices,qBOLD_test.shape[1],qBOLD_test.shape[2]))
error_array = np.zeros((2,n_slices,qBOLD_test.shape[1],qBOLD_test.shape[2]))
p0_array = np.concatenate((S0,R2,Y,DBV,chi_nb),axis=-1)
p0_array=p0_array[offset:offset+n_slices]
y_data = y_data[offset:offset+n_slices]
for x in tqdm(range(n_slices)):  #qBOLD_test.shape[0]  
    for y in range(qBOLD_test.shape[1]):
        for z in range(qBOLD_test.shape[2]):
            try:
                popt,pcov = curve_fit(qqBOLD_GESSE,x_data,y_data[x,y,z,:],p0_array[x,y,z,:],bounds=(physical_lower_bounds,physical_upper_bounds))
                popt_array[:,x,y,z]=popt
            except:
                error_array[0,x,y,z]=1
                pass
            try:
                popt_basic,pcov_basic = curve_fit(f_qBOLD_basic,x_data,y_data[x,y,z,:-1],p0=[0.5,15],bounds=([0.1,1],[1.5,30]))
                p1=np.array([popt_basic[0],popt_basic[1],0.5,0.05,0])
                p1 = check_start_params_for_physical_bounds(p1,physical_lower_bounds,physical_upper_bounds,minimal_space)
                popt2,pcov2 = curve_fit(qqBOLD_GESSE,x_data,y_data[x,y,z,:],p1,bounds=(physical_lower_bounds,physical_upper_bounds))
                popt2_array[:,x,y,z]=popt2
            except:
                error_array[1,x,y,z]=1
                pass



#%%
archive_name = "D:/Brain_Phantom/Patches_no_air_big_GESSE/15GB_0Pnoise_test_fit_perfect_prior_no_prior_first5000"
np.savez(archive_name,truth=p0_array,perfect_prior=popt_array,no_prior = popt2_array,failed_fits = error_array)


# %%
#code 40 seconds for one 30*30 slice

archive_name = "D:/Brain_Phantom/Patches_no_air_big_GESSE/15GB_0Pnoise_test_fit_perfect_prior_no_prior_first5000.npz"
Dataset=np.load(archive_name)
label = Dataset['truth']
pred_perf=Dataset['perfect_prior']
pred_no = Dataset['no_prior']

pred_perf=np.moveaxis(pred_perf,0,-1)
pred_no=np.moveaxis(pred_no,0,-1)
#%%
label_list = []
pred_perf_list = []
pred_no_list = []
for i in range(5):
    label_list.append(label[:,:,:,i].flatten())
    pred_perf_list.append(pred_perf[:,:,:,i].flatten())
    pred_no_list.append(pred_no[:,:,:,i].flatten())

#%%
QQplt.check_full_confusion_matrix_normed(label_list,pred_perf_list,'confusion_test_GESSE_0noise_fit_perfect_prior')
QQplt.check_full_confusion_matrix_normed(label_list,pred_no_list,'confusion_test_GESSE_0noise_fit_no_prior')

# %%
