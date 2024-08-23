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

def create_images(seg,directory_name,multiples,noise,step_size=10):
    t=np.array([29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91])/1000

    nda_seg = sitk.GetArrayViewFromImage(seg)
    patch_size = 30 # pixel
    n=find_number_of_patches(seg,patch_size,step_size)
    index = [*range(n*multiples)]
    np.random.shuffle(index)
 
    
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

                    S0_array = np.zeros((patch_size,patch_size,1,1),dtype=np.float32)
                    R2_array = np.zeros((patch_size,patch_size,1,1),dtype=np.float32)
                    Y_array = np.zeros((patch_size,patch_size,1,1),dtype=np.float32)
                    nu_array = np.zeros((patch_size,patch_size,1,1),dtype=np.float32)
                    chi_nb_array = np.zeros((patch_size,patch_size,1,1),dtype=np.float32)
                    input_array  = np.zeros((patch_size,patch_size,len(t)+1,1),dtype=np.float32)
                    

                    for xx in range(patch_size):
                        for yy in range(patch_size):
                            type = nda_seg_patch[xx,yy]
                            S0_array[xx,yy,:,0] = a[type]
                            R2_array[xx,yy,:,0] = b[type]
                            Y_array[xx,yy,:,0] = c[type]
                            nu_array[xx,yy,:,0] = d[type]
                            chi_nb_array[xx,yy,:,0] = e[type]
                            input_array[xx,yy,:-1,0]  = qBOLD[type,:]
                            input_array[xx,yy,-1,0]    = QSM[type]

                    if noise:
                        input_array[:,:,:-1,0] = input_array[:,:,:-1,0] + rng.normal(loc=0,scale=1./100,size=input_array[:,:,:-1,0].shape)
                        input_array[:,:,-1,0]   = input_array[:,:,-1,0]    + rng.normal(loc=0,scale=0.1/100,size=input_array[:,:,-1,0] .shape)

                    num_digits=7
                    archive_name = f"{directory_name}{index[count]:0{num_digits}d}"
                    np.savez(archive_name,qqBOLD=input_array,S0=S0_array,R2=R2_array,Y=Y_array,nu=nu_array,chi_nb=chi_nb_array)

                    count += 1

    
#%%
directory = "../Brain_Phantom/Patches_no_air_big_GESSE_1pNoise_single_files_train_val_1_input/"
#Create train and test data separately by adjusting multiples, for example 9 for train and 1 for test
create_images(seg,directory,multiples=3,noise=True)
#%%
directory = "../Brain_Phantom/Patches_no_air_big_GESSE_1pNoise_single_files_test_1_input/"
create_images(seg,directory,multiples=1,noise=True,step_size=15)
#create_images(seg,"",multiples=9,noise=False)
#create_images(seg,"",multiples=1,noise=False)

#42441    *    (5 + 16 + 1)        *30*30    *  4                                  *20
#patches *(params + qBOLD + QSM) *image size *float32 = 3.361.327.200 = 3.35GB     67.226.544.000

#%%
