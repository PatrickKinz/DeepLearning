import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, randn, random, uniform
from numpy.random import default_rng,shuffle
rng=default_rng()
from tqdm import tqdm  #for progress bar
import SimpleITK as sitk
import QSM_and_qBOLD_functions as QQ
from tqdm import tqdm  #for progress bar


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
def create_whole_brain(M):
    for m in range(M):
        """ create parameter values for each tissue type """
        #currently assuming same parameter distribution for all tissue types
        N_tissues=17 #Air and Abnormal_WM not included
        t=np.array([29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91])/1000
        #t=np.array([2.72,8.80,13.00,17.20,21.40])/1000
        #t=np.array([2.72,8.80,13.00,17.20,21.40,25.81,29.81,3384,37.84])/1000

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
        signal = QQ.f_qBOLD_GESSE(S0,R2,Y,nu,chi_nb,t)
        print('signal', signal.shape)
        QSM = QQ.f_QSM(Y,nu,chi_nb)
        print('QSM',QSM.shape)

        """ Save Parameters as truth/targets and signal as input/targets for CNN
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
         """
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


def threshold_based_crop(image):
    """From SITK 70_Data_Augmentation.ipynb
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.
    """
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute( sitk.OtsuThreshold(image, inside_value, outside_value) )
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
#%%
def examples_for_cropping():
    seg_cropped = threshold_based_crop(seg)
    nda_seg_cropped = sitk.GetArrayViewFromImage(seg_cropped)
    nda_seg_cropped.shape
    plt.figure()
    plt.imshow(nda_seg_cropped[2,:,:])
    plt.colorbar()

    seg_slice_cropped = threshold_based_crop(sitk.GetImageFromArray(sitk.GetArrayViewFromImage(seg)[2,:,:]))
    nda_seg_slice_cropped = sitk.GetArrayViewFromImage(seg_slice_cropped)
    nda_seg_slice_cropped.shape
    plt.figure()
    plt.imshow(nda_seg_slice_cropped)
    plt.colorbar()

examples_for_cropping()
#%%

""" Pull M random slices from 3D matrix and save them (total number M*N)"""
""" Augment data total number ?*M*N """
#Rotation in each Ebene
#Varied cuts. Cor, Sag, Trans or even in between?
#Varied size?
#Deformation?

def find_number_of_patches(seg,patch_size,step_size):
    n=0
    for i in tqdm(range(seg.GetDepth())):
        seg_slice = threshold_based_crop(sitk.GetImageFromArray(nda_seg[i,:,:]))
        nda_seg_slice = sitk.GetArrayFromImage(seg_slice)
        for x in range(0,nda_seg_slice.shape[0]-patch_size,step_size):
            for y in range(0,nda_seg_slice.shape[1]-patch_size,step_size):
                n=n+1
    return n

n=find_number_of_patches(seg,30,10)
print(n)

n=find_number_of_patches(seg,30,15)
print(n)
#%%

def create_images(seg,archive_name,multiples,noise,step_size=10):
    t=np.array([29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91])/1000

    nda_seg = sitk.GetArrayViewFromImage(seg)
    patch_size = 30 # pixel
    n=find_number_of_patches(seg,patch_size,step_size)
    index = [*range(n*multiples)]
    np.random.shuffle(index)
    S0_array = np.zeros((n*multiples,patch_size,patch_size,1),dtype=np.float32)
    R2_array = np.zeros((n*multiples,patch_size,patch_size,1),dtype=np.float32)
    Y_array = np.zeros((n*multiples,patch_size,patch_size,1),dtype=np.float32)
    nu_array = np.zeros((n*multiples,patch_size,patch_size,1),dtype=np.float32)
    chi_nb_array = np.zeros((n*multiples,patch_size,patch_size,1),dtype=np.float32)
    qBOLD_array  = np.zeros((n*multiples,patch_size,patch_size,len(t)),dtype=np.float32)
    QSM_array    = np.zeros((n*multiples,patch_size,patch_size,1),dtype=np.float32)
    
    count = 0

    N_tissues=18 #Abnormal_WM not included

    for i in tqdm(range(seg.GetDepth())): #seg.GetDepth()
        #crop empty area around brain
        seg_slice = threshold_based_crop(sitk.GetImageFromArray(nda_seg[i,:,:]))
        nda_seg_slice = sitk.GetArrayFromImage(seg_slice)
        for x in range(0,nda_seg_slice.shape[0]-patch_size,step_size):
            for y in range(0,nda_seg_slice.shape[1]-patch_size,step_size):
                nda_seg_patch = nda_seg_slice[x:x+patch_size,y:y+patch_size]
                #print(x,' ',y,' ', count)
                """generate parameters"""
                #currently assuming same parameter distribution for all tissue types

                for j in range(multiples):
                    #a=0.5*np.ones(N_tissues) #S0 # TODO: Allow some variation
                    a=uniform(0.2,1.0,N_tissues)
                    b=random(N_tissues) #R2
                    c=random(N_tissues) #Y
                    d=random(N_tissues) #nu
                    e=random(N_tissues) #chi_nb

                    """ Calculate qBOLD and QSM for each tissue type """
                    S0 = a   #S0     = 1000 + 200 * randn(N).T
                    R2 = (30-1) * b + 1  #from 1 to 30
                    SaO2 = 0.98
                    Y  = (SaO2 - 0.01) * c + 0.01   #from 1% to 98%
                    nu = (0.1 - 0.001) * d + 0.001  #from 0.1% to 10%
                    chi_nb = ( 0.1-(-0.1) ) * e - 0.1 #from -0.1 ppb to 0.1 ppb
                    
                    """calculate qBOLD and QSM """
                    qBOLD = QQ.f_qBOLD_GESSE(S0,R2,Y,nu,chi_nb,t)
                    QSM   = QQ.f_QSM(Y,nu,chi_nb)


                    for xx in range(patch_size):
                        for yy in range(patch_size):
                            type = nda_seg_patch[xx,yy]
                            S0_array[index[count],xx,yy,:] = a[type]
                            R2_array[index[count],xx,yy,:] = b[type]
                            Y_array[index[count],xx,yy,:] = c[type]
                            nu_array[index[count],xx,yy,:] = d[type]
                            chi_nb_array[index[count],xx,yy,:] = e[type]
                            qBOLD_array[index[count],xx,yy,:]  = qBOLD[type,:]
                            QSM_array[index[count],xx,yy,0]    = QSM[type]

                    if noise:
                        qBOLD_array[index[count],:,:,:] = qBOLD_array[index[count],:,:,:] + rng.normal(loc=0,scale=1./100,size=qBOLD_array[index[count],:,:,:].shape)
                        QSM_array[index[count],:,:,:]   = QSM_array[index[count],:,:,:]   + rng.normal(loc=0,scale=0.1/100,size=QSM_array[index[count],:,:,:].shape)

                    count += 1

    np.savez(archive_name,qBOLD=qBOLD_array,QSM=QSM_array,S0=S0_array,R2=R2_array,Y=Y_array,nu=nu_array,chi_nb=chi_nb_array)

#%%
#Create train and test data separately by adjusting multiples, for example 9 for train and 1 for test
create_images(seg,"../Brain_Phantom/Patches_no_air_big_GESSE/15GB_0Pnoise_train_val",multiples=2,noise=False)
#%%
create_images(seg,"../Brain_Phantom/Patches_no_air_big_GESSE/15GB_0Pnoise_test",multiples=1,noise=False,step_size=15)
#create_images(seg,"",multiples=9,noise=False)
#create_images(seg,"",multiples=1,noise=False)

#42441    *    (5 + 16 + 1)        *30*30    *  4                                  *20
#patches *(params + qBOLD + QSM) *image size *float32 = 3.361.327.200 = 3.35GB     67.226.544.000

#%%
""" Loop over saved slices """
    """ Add noise to signal slices, repeat K times (total number K*?*M*N)"""

    """ Norm signal slices """

    """ Save noisy normed slices """

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
            qBOLD[i,:,:,:] = Dataset['qBOLD'] + rng.normal(loc=0,scale=1./100,size=Dataset['qBOLD'].shape)
            QSM[i,:,:,:]   = Dataset['QSM']   + rng.normal(loc=0,scale=0.1/100,size=Dataset['QSM'].shape)
        else:
            qBOLD[i,:,:,:] = Dataset['qBOLD']
            QSM[i,:,:,:]   = Dataset['QSM']

    np.savez(archive_name,qBOLD=qBOLD,QSM=QSM,S0=S0,R2=R2,Y=Y,nu=nu,chi_nb=chi_nb)
