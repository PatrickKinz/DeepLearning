import numpy as np
from tqdm import tqdm
from numpy.random import default_rng,shuffle
rng=default_rng()

filenames=[]
filenumber=int(848820)
for count in range(filenumber):
    filenames.append("{0}.npz".format(count).zfill(6+4))
filenames_shuffled=shuffle(filenames)
#split
threshold1 = int(filenumber*0.9/20*5)
threshold2 = int(filenumber/20*5)
filenames_train_shuffled=filenames[:threshold1]
filenames_test_shuffled=filenames[threshold1:threshold2]
len(filenames_train_shuffled)
len(filenames_test_shuffled)

#Params.shape#(1, 30, 30, 5)
#qBOLD.shape#(1, 30, 30, 16)
#QSM.shape#(1, 30, 30, 1)

def make_archive(filenames,archive_name,noise):
    Dataset=np.load("../Brain_Phantom/Patches_no_air_big/NumpyArchives/NumpyArchiv_000000.npz")
    qBOLD=np.zeros((len(filenames),Dataset['qBOLD'].shape[1],Dataset['qBOLD'].shape[2],Dataset['qBOLD'].shape[3]   ),dtype=np.float32)
    QSM=np.zeros((len(filenames),Dataset['QSM'].shape[1],Dataset['QSM'].shape[2],Dataset['QSM'].shape[3]   ),dtype=np.float32)
    S0=np.zeros((len(filenames),Dataset['Params'].shape[1],Dataset['Params'].shape[2]  ),dtype=np.float32)
    R2=np.zeros((len(filenames),Dataset['Params'].shape[1],Dataset['Params'].shape[2]   ),dtype=np.float32)
    Y=np.zeros((len(filenames),Dataset['Params'].shape[1],Dataset['Params'].shape[2]   ),dtype=np.float32)
    nu=np.zeros((len(filenames),Dataset['Params'].shape[1],Dataset['Params'].shape[2]   ),dtype=np.float32)
    chi_nb=np.zeros((len(filenames),Dataset['Params'].shape[1],Dataset['Params'].shape[2]   ),dtype=np.float32)

    for i in tqdm(range(len(filenames))):
        Dataset=np.load("../Brain_Phantom/Patches_no_air_big/NumpyArchives/NumpyArchiv_" + filenames_train_shuffled[i])
        S0[i,:,:]=Dataset['Params'][:,:,:,0]
        R2[i,:,:]=Dataset['Params'][:,:,:,1]
        Y[i,:,:]=Dataset['Params'][:,:,:,2]
        nu[i,:,:]=Dataset['Params'][:,:,:,3]
        chi_nb[i,:,:]=Dataset['Params'][:,:,:,4]
        if noise:
            qBOLD[i,:,:,:]=Dataset['qBOLD'] + rng.normal(loc=0,scale=1./100,size=Dataset['qBOLD'].shape)
            QSM[i,:,:,:]=Dataset['QSM'] + rng.normal(loc=0,scale=0.1/100,size=Dataset['QSM'].shape)
        else:
            qBOLD[i,:,:,:]=Dataset['qBOLD']
            QSM[i,:,:,:]=Dataset['QSM']

    np.savez(archive_name,qBOLD=qBOLD,QSM=QSM,S0=S0,R2=R2,Y=Y,nu=nu,chi_nb=chi_nb)


make_archive(filenames_train_shuffled,"../Brain_Phantom/Patches_no_air_big/15GB_1Pnoise_train_val",noise=True)
make_archive(filenames_test_shuffled,"../Brain_Phantom/Patches_no_air_big/15GB_1Pnoise_test",noise=True)
make_archive(filenames_train_shuffled,"../Brain_Phantom/Patches_no_air_big/15GB_0noise_train_val",noise=False)
make_archive(filenames_test_shuffled,"../Brain_Phantom/Patches_no_air_big/15GB_0noise_test",noise=False)
