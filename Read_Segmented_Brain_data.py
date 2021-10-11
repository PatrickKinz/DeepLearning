import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

inputImageFolder = "C:/Users/pk24/Documents/Programming/Brain_Phantom/NV_1_NV_1T_20210902153453_SEED_0/MODEL_NV_1_148_SLICES"
inputImageFileNameList=[]
for i in range(1,10):
    inputImageFileNameList.append(inputImageFolder+'/SLICE_00' +str(i)+'.TIF')
#inputImageFileNameList
for i in range(10,100):
    inputImageFileNameList.append(inputImageFolder+'/SLICE_0' +str(i)+'.TIF')
#inputImageFileNameList[0:20]
for i in range(100,148+1):
    inputImageFileNameList.append(inputImageFolder+'/SLICE_' +str(i)+'.TIF')
#inputImageFileNameList[-20:]

image = sitk.ReadImage(inputImageFileNameList, imageIO="TIFFImageIO")
print(image)
#%%
nda = sitk.GetArrayViewFromImage(image)
nda.shape
# (148, 256, 256, 3)
n_z = 70
plt.imshow(nda[n_z,:,:,:])
plt.axis('off')
"""
MODEL_NV_1_148_SLICES
Tissue_Name                 ,Volume_(cc)    ,Color_code_(RGB)
GM                          ,854.86         ,(127,127,127)
WM                          ,572.62         ,(255,255,255)
CSF                         ,182.45         ,(000,000,180)
Pallidus                    ,3.99           ,(255,000,000)
Putamen                     ,9.73           ,(000,120,000)
Thalamus                    ,13.95          ,(080,120,080)
Caudatus                    ,10.35          ,(000,110,000)
Nigra                       ,1.19           ,(200,000,000)
Red_Nucleus                 ,0.66           ,(255,050,000)
Dentate_Nucleus             ,1.67           ,(128,255,000)
Low_PD                      ,825.68         ,(190,000,255)
Fat                         ,409.18         ,(255,190,190)
Muscle                      ,552.09         ,(128,000,000)
Vitreous_Humor              ,14.25          ,(000,000,255)
Extra_Cranial_Connective    ,54.20          ,(127,127,255)
Extra_Cranial_Fluid         ,16.54          ,(000,180,180)
Intra_Cranial_Connective    ,20.04          ,(128,080,255)
Abnormal_WM                 ,0.00           ,(255,255,000)
"""

list_Color_code = [  (127,127,127)
                    ,(255,255,255)
                    ,(  0,  0,180)
                    ,(255,  0,  0)
                    ,(  0,120,  0)
                    ,( 80,120, 80)
                    ,(  0,110,  0)
                    ,(200,  0,  0)
                    ,(255, 50,  0)
                    ,(128,255,  0)
                    ,(190,  0,255)
                    ,(255,190,190)
                    ,(128,  0,  0)
                    ,(  0,  0,255)
                    ,(127,127,255)
                    ,(  0,180,180)
                    ,(128, 80,255)
                    ,(255,255,  0)
                  ]


# %%
nda.shape
nda.shape[:3]
nda[60,60,60,:]
list_Color_code[10]
nda[60,60,60,:] == list_Color_code[10]
np.all(nda[60,60,60,:] == list_Color_code[10])
segments = np.zeros(nda.shape[:3])
for x in range(nda.shape[0]):
     for y in range(nda.shape[1]):
         for z in range(nda.shape[2]):
             for i in range(len(list_Color_code)):
                 if( np.all(nda[x,y,z,:] == list_Color_code[i]) ):
                    segments[x,y,z] = i+1

#nda[0,0,0,:]
#list_Color_code[17]
# %%
plt.imshow(segments[70,:,:])
plt.axis('off')
plt.colorbar()

# %%
segments=np.cast['uint8'](segments)
segments[0,0,0]
img_segmentation =  sitk.GetImageFromArray(segments)
img_segmentation.CopyInformation(image)

#print(image)
#print(img_segmentation)
# %%
""" Two ways how to write a file, compression supposed to work well for labels """
writer = sitk.ImageFileWriter()
writer.SetImageIO("TIFFImageIO")
writer.SetFileName("C:/Users/pk24/Documents/Programming/Brain_Phantom/Segmentation.TIF")
writer.Execute(img_segmentation)

sitk.WriteImage(img_segmentation,"C:/Users/pk24/Documents/Programming/Brain_Phantom/Segmentation.TIF" ,useCompression=True)
# %%
""" Test writing was correct"""

img_reload = sitk.ReadImage("C:/Users/pk24/Documents/Programming/Brain_Phantom/Segmentation.TIF")
nda_reload = sitk.GetArrayViewFromImage(img_reload)
nda_reload.shape
nda_reload[0,0,0]
print(img_reload)

plt.imshow(nda_reload[70,:,:])
plt.axis('off')
plt.colorbar()
# %% QMCI

inputQMCIFolder = "C:/Users/pk24/Documents/Programming/Brain_Phantom/NV_1_NV_1T_20210902153453_SEED_0/QMCI"
inputQMCIFileNameList=[]
for i in range(104,248+1,4):
    inputQMCIFileNameList.append(inputQMCIFolder+'/QT' +str(i)+'.TIF')
inputQMCIFileNameList

QMCI = sitk.ReadImage(inputQMCIFileNameList, imageIO="TIFFImageIO")

npaQMCI = sitk.GetArrayViewFromImage(QMCI)
npaQMCI.shape
# (148, 256, 256, 3)
n_z = 18
# %%
plt.imshow(npaQMCI[n_z,:,:,:])
plt.axis('off')


# %% R1
plt.imshow(npaQMCI[n_z,:,:,0])
plt.axis('off')

# %% R2
plt.imshow(npaQMCI[n_z,:,:,1])
plt.axis('off')

# %% PD
plt.imshow(npaQMCI[n_z,:,:,2])
plt.axis('off')
