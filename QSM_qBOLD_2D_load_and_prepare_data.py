# %% import modules
import numpy as np
import h5py
from numpy.random import rand, randn
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm  #for progress bar
#%%
#data_dir = "C:/Users/pk24/Documents/Programming/Brain_Phantom/Patches/"
data_dir = "../Brain_Phantom/Patches/"
def read_data(data_dir):
    file_list_Params=[]
    file_list_qBOLD=[]
    file_list_QSM=[]
    for i in tqdm(range(100)): #42441
        file_number = "{0}".format(i).zfill(6)
        file_list_Params.append(data_dir+ "Params/Params_"+file_number+ ".TIF")
        file_list_qBOLD.append(data_dir + "qBOLD/qBOLD_"+file_number+ ".TIF")
        file_list_QSM.append(data_dir + "QSM/QSM_"+file_number+ ".TIF")
    #print(file_list_Params)
    Params = sitk.GetArrayFromImage(sitk.ReadImage(file_list_Params))
    #print(Params.shape)
    Params = np.moveaxis(Params,1,-1) #channels last
    #print(Params.shape)
    qBOLD  = sitk.GetArrayFromImage(sitk.ReadImage(file_list_qBOLD))
    qBOLD  = np.moveaxis(qBOLD,1,-1) #channels last
    QSM  = sitk.GetArrayFromImage(sitk.ReadImage(file_list_QSM))
    QSM= np.expand_dims(QSM,axis=-1) #add one channel
    return Params,qBOLD,QSM

Params,qBOLD,QSM = read_data(data_dir)
print(Params.shape)
print(qBOLD.shape)
print(QSM.shape)
#%%
"""
print(np.any(~np.isfinite(Params)))
print(np.isfinite(Params))

print(Params[0,0,0,:])#[3.783506e-44 2.382207e-44 1.401298e-45 7.917653e+05 4.624285e-44]
print(Params[0,0,1,:])#[       0.        0.        0. 50675840.        0.]
print(Params[0,0,2,:])#[          nan           nan 7.0356876e-09 5.2821254e+04           nan]
print(Params[0,:,:,0])
#%%
plt.figure()
plt.imshow(Params[0,:,:,0])
plt.colorbar()
Params[0,0,0:10,0]
#%%

Params_test = sitk.GetArrayFromImage(sitk.ReadImage("../Brain_Phantom/Patches/Params/Params_000000.TIF"))
Params_test.shape
Params_test = np.moveaxis(Params_test,0,-1)
Params_test.shape
#%%
plt.figure()
plt.imshow(Params_test[:,:,0])
plt.colorbar()

Params_test[0,0:10,0]
"""
#%% shuffle images

def shuffle_data(Params,qBOLD,QSM):
    idx = rand(Params.shape[0]).argsort()
    idx.shape
    #print(idx[0])
    Params_shuffled = np.zeros(Params.shape)
    qBOLD_shuffled = np.zeros(qBOLD.shape)
    QSM_shuffled = np.zeros(QSM.shape)
    for i in tqdm(range(len(idx))):
        Params_shuffled[i,:,:,:] = Params[idx[i],:,:,:]
    for i in tqdm(range(len(idx))):
        qBOLD_shuffled[i,:,:,:] = qBOLD[idx[i],:,:,:]
    for i in tqdm(range(len(idx))):
        QSM_shuffled[i,:,:,:] = QSM[idx[i],:,:,:]
    return Params_shuffled,qBOLD_shuffled,QSM_shuffled

#%%

def add_noise_qBOLD(a,mu,sigma):
    #qBOLD signal is stored with amplitude 0.5, only positive numbers
    noise = np.random.normal(loc=mu,scale=sigma,size=a.shape)
    a = a + noise
    return np.maximum(a,np.zeros(a.shape))

def add_noise_QSM(a,mu,sigma):
    #QSM is close to zero, positive and negative
    noise = np.random.normal(loc=mu,scale=sigma,size=a.shape)
    a = a + noise
    return a

def add_noise_data(Params,qBOLD,QSM):
    #Params are truth and dont get any noise
    qBOLD = add_noise_qBOLD(qBOLD,0,0.01) #signal 0.5
    QSM = add_noise_QSM(QSM,0,0.001) #dont know the QSM signal level
    return Params,qBOLD,QSM

#%%

def norm_qBOLD(a):

    return a

def norm_data(Params,qBOLD,QSM):
    #Params are already 0 to 1
    qBOLD = norm_qBOLD(qBOLD)
    #QSM is already -1 to +1 or even less
    return Params,qBOLD,QSM


#%% split in training and test
def split_training_test(Params,qBOLd,QSM,percentage=0.9):
    threshold = int(Params.shape[0]*percentage)
    Params_training= Params[:threshold,:,:,:]
    Params_test= Params[threshold:,:,:,:]
    qBOLD_training= qBOLD[:threshold,:,:,:]
    qBOLD_test= qBOLD[threshold:,:,:,:]
    QSM_training= QSM[:threshold,:,:,:]
    QSM_test= QSM[threshold:,:,:,:]
    return Params_training,Params_test,qBOLD_training,qBOLD_test,QSM_training,QSM_test

# %% norm
"""
def norm_signal_array(input,target):
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            for k in range(input.shape[2]):
                scale = input[i,j,k,0]
                input[i,j,k,:] =  input[i,j,k,:]/scale
                target[i,j,k,0] =  target[i,j,k,0]/scale
    return input,target
input_noise_norm_train, target_train = norm_signal_array(input_noise_train, target_train)
plt.plot(input_noise_norm_train[0,1,1,:],'o')
input_noise_norm_test, target_test = norm_signal_array(input_noise_test, target_test)
"""
"""
input_noise_norm_train = np.log(input_noise_train)
input_noise_norm_test = np.log(input_noise_test)
input_norm_test = np.log(input_test)

target_train[:,:,:,0] = np.log(target_train[:,:,:,0])
target_test[:,:,:,0] = np.log(target_test[:,:,:,0])
"""
"""
input_noise_norm_train = input_noise_train
input_noise_norm_test = input_noise_test
input_norm_test = input_test
"""
# %% augment data
"""
def augment_data(m):
    m = np.concatenate((m,np.flip(m, axis=1)), axis=0)
    m = np.concatenate((m,np.rot90(m, k=1, axes=(2,1))), axis=0)
    m = np.concatenate((m,np.flip(m, axis=(1,2))), axis=0)
    return m


#input_train = augment_data(input_train)
input_noise_norm_train = augment_data(input_noise_norm_train)
getsizeof(input_noise_norm_train)/(1000*1000*1000) # GB
target_train = augment_data(target_train)

input_test = augment_data(input_test)
input_noise_norm_test = augment_data(input_noise_norm_test)
target_test = augment_data(target_test)


# %% Look at rotation
fig, axes = plt.subplots(nrows=2, ncols=2)
ax = axes.ravel()
ax[0].imshow(input_noise_norm_test[0,:,:,1], cmap='Greys')
ax[1].imshow(input_noise_norm_test[100,:,:,1], cmap='Greys')
ax[2].imshow(input_noise_norm_test[200,:,:,1], cmap='Greys')
ax[3].imshow(input_noise_norm_test[300,:,:,1], cmap='Greys')
fig, axes = plt.subplots(nrows=2, ncols=2)
ax = axes.ravel()
ax[0].imshow(input_noise_norm_test[400,:,:,1], cmap='Greys')
ax[1].imshow(input_noise_norm_test[500,:,:,1], cmap='Greys')
ax[2].imshow(input_noise_norm_test[600,:,:,1], cmap='Greys')
ax[3].imshow(input_noise_norm_test[700,:,:,1], cmap='Greys')




"""
def load_and_prepare_data():
    Params,qBOLD,QSM = read_data()
    Params,qBOLD,QSM = shuffle_data(Params,qBOLD,QSM)
    Params,qBOLD,QSM = add_noise_data(Params,qBOLD,QSM)
    Params,qBOLD,QSM = norm_data(Params,qBOLD,QSM)
    Params_training,Params_test,qBOLD_training,qBOLD_test,QSM_training,QSM_test = split_training_test(Params,qBOLD,QSM)
    return Params_training,Params_test,qBOLD_training,qBOLD_test,QSM_training,QSM_test
