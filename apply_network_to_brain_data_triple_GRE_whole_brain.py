#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sp
import nibabel as nib

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
from tensorflow.keras import layers

import QSM_qBOLD_2D_plotting_functions as QQplt
import QSM_and_qBOLD_functions as QQfunc
import os
from tqdm import tqdm
#%%
# load niftis
subject_name = "RC1155_sub-187"
path_name1 = "D:/Neurologie/QORMIND/PK/"
#RC1121_sub-186
path_name2 = "/Nifti/ses-mri01/anat/"
#8d
path_name3 = "/output/QSM_MEDI/"


img_mag_8d = nib.load("".join([path_name1,subject_name,path_name2,"8d",path_name3,subject_name,"_ses-mri01_8d_QSM_MEDI_part-mag.nii.gz"]))
img_mag_18d = nib.load("".join([path_name1,subject_name,path_name2,"18d",path_name3,subject_name,"_ses-mri01_18d_QSM_MEDI_part-mag.nii.gz"]))
img_mag_30d = nib.load("".join([path_name1,subject_name,path_name2,"30d",path_name3,subject_name,"_ses-mri01_30d_QSM_MEDI_part-mag.nii.gz"]))
img_mask_8d = nib.load("".join([path_name1,subject_name,path_name2,"8d",path_name3,subject_name,"_ses-mri01_8d_QSM_MEDI_mask_QSM.nii.gz"]))
#img_mask_18d = nib.load("".join([path_name1,subject_name,path_name2,"18d",path_name3,subject_name,"_ses-mri01_18d_QSM_MEDI_mask_QSM.nii.gz"]))
#img_mask_30d = nib.load("".join([path_name1,subject_name,path_name2,"30d",path_name3,subject_name,"_ses-mri01_30d_QSM_MEDI_mask_QSM.nii.gz"]))
img_chi_8d = nib.load("".join([path_name1,subject_name,path_name2,"8d",path_name3,subject_name,"_ses-mri01_8d_QSM_MEDI_Chimap.nii.gz"]))
img_chi_18d = nib.load("".join([path_name1,subject_name,path_name2,"18d",path_name3,subject_name,"_ses-mri01_18d_QSM_MEDI_Chimap.nii.gz"]))
img_chi_30d = nib.load("".join([path_name1,subject_name,path_name2,"30d",path_name3,subject_name,"_ses-mri01_30d_QSM_MEDI_Chimap.nii.gz"]))
mag_8d=img_mag_8d.get_fdata()
mag_18d=img_mag_18d.get_fdata()
mag_30d=img_mag_30d.get_fdata()
Mask_8d=img_mask_8d.get_fdata()
#Mask_18d=img_mask_18d.get_fdata()
#Mask_30d=img_mask_30d.get_fdata()
chi = np.stack([img_chi_8d.get_fdata(),img_chi_18d.get_fdata(),img_chi_30d.get_fdata()],axis=-1)
#%%
""" mag_gesse and QSM_gesse as input for Neural network"""

n_slice = 42
crop_x=0#40 #40
crop_y=0#40# 30
size_x = 480-2*crop_x#30
size_y = 640-4*crop_y#size_x
plt.figure()
plt.imshow(mag_8d[crop_x:crop_x+size_x,crop_y:crop_y+size_y,n_slice,0])
plt.axis('off')
plt.show()


plt.figure()
plt.imshow(chi[crop_x:crop_x+size_x,crop_y:crop_y+size_y,n_slice,1])
plt.axis('off')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Mask_8d[crop_x:crop_x+size_x,crop_y:crop_y+size_y,n_slice])
plt.axis('off')
plt.colorbar()
plt.show()
#%%
#apply crop
mag_8d_cor = mag_8d[crop_x:crop_x+size_x,crop_y:crop_y+size_y,:,:]
mag_18d_cor = mag_18d[crop_x:crop_x+size_x,crop_y:crop_y+size_y,:,:]
mag_30d_cor = mag_30d[crop_x:crop_x+size_x,crop_y:crop_y+size_y,:,:]

QSM = chi[crop_x:crop_x+size_x,crop_y:crop_y+size_y,:,:]
Mask_8d = Mask_8d[crop_x:crop_x+size_x,crop_y:crop_y+size_y,:]

mag_8d_rearr = np.moveaxis(mag_8d_cor,2,0)
mag_18d_rearr = np.moveaxis(mag_18d_cor,2,0)
mag_30d_rearr = np.moveaxis(mag_30d_cor,2,0)

QSM_array = np.moveaxis(QSM,2,0)
Mask_array = np.moveaxis(Mask_8d,2,0)
max_8d = np.max(mag_8d_rearr)
max_18d = np.max(mag_18d_rearr)
max_30d = np.max(mag_30d_rearr)
max_total = np.max([max_8d,max_18d,max_30d])
qBOLD_array_8d = mag_8d_rearr/max_total
qBOLD_array_18d = mag_18d_rearr/max_total
qBOLD_array_30d = mag_30d_rearr/max_total

print(qBOLD_array_8d.shape)
print(QSM_array.shape)
print(Mask_array.shape)
#add more axis to mask

Mask_array = np.expand_dims(Mask_array,-1)
#Mask_array = np.expand_dims(Mask_array,-1)
print(qBOLD_array_8d.shape)
print(QSM_array.shape)
print(Mask_array.shape)



# %%
version = "no_air_1Pnoise_15GB_triple_GRE/"

model_params = keras.models.load_model("models/"+version+ "Model_2D_image_triple_GRE_1Pnoise_all3D.h5")
model_params.summary()
#%%
keras.utils.plot_model(model_params, show_shapes=True)

#%%
prediction = []
for j in range(6):
    prediction.append(np.zeros(Mask_array.shape))
for i in tqdm(range(qBOLD_array_8d.shape[0])):
    #print(i)
    #select only one slice
    qBOLD_array_8d_temp = qBOLD_array_8d[i,:,:,:]
    qBOLD_array_18d_temp = qBOLD_array_18d[i,:,:,:]
    qBOLD_array_30d_temp = qBOLD_array_30d[i,:,:,:]
    QSM_array_temp = QSM_array[i,:,:,:]
    Mask_array_temp = Mask_array[i,:,:,:]

    qBOLD_array_8d_temp = np.expand_dims(qBOLD_array_8d_temp,0) #if only one slice selected
    qBOLD_array_18d_temp = np.expand_dims(qBOLD_array_18d_temp,0) #if only one slice selected
    qBOLD_array_30d_temp = np.expand_dims(qBOLD_array_30d_temp,0) #if only one slice selected
    QSM_array_temp = np.expand_dims(QSM_array_temp,0)
    Mask_array_temp = np.expand_dims(Mask_array_temp,0)
    
    p = model_params.predict([qBOLD_array_8d_temp,qBOLD_array_18d_temp,qBOLD_array_30d_temp,QSM_array_temp])
    for j in range(len(p)):
        prediction[j][i,:,:,:]=p[j][:,:,:,0]
    

#%%

#label_transformed=QQplt.translate_Params(test_list)
prediction_transformed=QQplt.translate_Params_T1(prediction)
#%%
#save as nifti
names = ['S0.nii.gz','R2.nii.gz','Y.nii.gz','DBV.nii.gz','chi_nb.nii.gz','T1.nii.gz']
for i in tqdm(range(len(prediction_transformed))):
    pred_temp = np.squeeze(prediction_transformed[i]*Mask_array)
    #print(pred_temp.shape)
    pred_temp = np.moveaxis(pred_temp,0,2)
    #print(pred_temp.shape)
    img=nib.Nifti1Image(pred_temp,img_mask_8d.affine)
    nib.save(img,os.path.join("".join([path_name1,subject_name,'/Nifti/ses-mri01/anat/OEF']),names[i]))
    

#%%
prediction_squeezed = [] #shape (1,x,y)
prediction_squeezed_larger = [] #shape (1,x,y,1)
for i in range(len(prediction_transformed)):
    prediction_squeezed.append( np.expand_dims(np.squeeze(prediction_transformed[i]),0) )
    prediction_squeezed_larger.append(np.expand_dims(prediction_squeezed[i],-1))
#%%
Number=0
QQplt.check_Params_transformed(prediction_squeezed,prediction_squeezed_larger,Number,'CNN_2D_image_triple_GRE_1Pnoise_all3D_real_data')

#%%


def plot_prediction(image,mask):
    fig, axes = plt.subplots(nrows=1, ncols=6,figsize=(12,2))
    ax = axes.ravel()
    P0=ax[0].imshow(np.rot90(np.squeeze(image[0])*mask),cmap='gray')
    P0.set_clim(.0,1.2)
    ax[0].title.set_text('$S_0$ [a.u.]')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    plt.colorbar(P0,ax=ax[0])
    P1=ax[1].imshow(np.rot90(np.squeeze(image[1])*mask),cmap='inferno')
    P1.set_clim(0,30)
    ax[1].title.set_text('$R_2$ [Hz]')
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    plt.colorbar(P1,ax=ax[1])
    P2=ax[2].imshow(np.rot90((1-np.squeeze(image[2])/0.98)*mask*100),cmap='inferno')
    P2.set_clim(.0,1*100)
    ax[2].title.set_text('OEF [%]')
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    plt.colorbar(P2,ax=ax[2])
    P3=ax[3].imshow(np.rot90(np.squeeze(image[3])*mask*100),cmap='inferno')
    P3.set_clim(0,0.1*100)
    ax[3].title.set_text('$v$ [%]')
    ax[3].get_xaxis().set_visible(False)
    ax[3].get_yaxis().set_visible(False)
    plt.colorbar(P3,ax=ax[3])
    P4=ax[4].imshow(np.rot90(np.squeeze(image[4])*mask*1000),cmap='gray')
    P4.set_clim(-.1*1000,.1*1000)
    ax[4].title.set_text('$\chi_{nb}$ [ppb]')
    ax[4].get_xaxis().set_visible(False)
    ax[4].get_yaxis().set_visible(False)
    plt.colorbar(P4,ax=ax[4])
    P5=ax[5].imshow(np.rot90(np.squeeze(image[5])*mask),cmap='gray')
    P5.set_clim(0,2)
    ax[5].title.set_text('$T_1$ [s]')
    ax[5].get_xaxis().set_visible(False)
    ax[5].get_yaxis().set_visible(False)
    plt.colorbar(P5,ax=ax[5])
    plt.show()
    fig.savefig('plots/real_brain_triple_GRE_all_3D_slice_42.png')
    
plot_prediction(prediction_transformed,np.squeeze(Mask_array)) 

# %%




"""
#%%
def change_model(model, new_input_shape):
    model.layers[0].batch_input_shape = new_input_shape

    new_model = keras.models.model_from_json(model.to_json())

    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            print("Loaded layer {}".format(layer.name))
        except:
            print("Could not transer weight for layer {}".format(layer.name))
    return new_model


#%%

#try to replace reshape layer
#model_params.layers
layer_names = [layer.name for layer in model_params.layers]
layer_idx = layer_names.index("Input_qBOLD")
qBOLD_in = model_params.layers[layer_idx].input
layer_idx = layer_names.index("spatial_dropout3d_3")
qBOLD_out = model_params.layers[layer_idx].output
#reshape_qBOLD_new = keras.layers.Reshape((128,96,128))(qBOLD_out)

model_qBOLD = keras.Model(
                inputs=qBOLD_in,
                outputs = qBOLD_out)

keras.utils.plot_model(model_qBOLD,"Model_qBOLD.png", show_shapes=True)

model_qBOLD_reshaped = change_model(model_qBOLD,(None,128,96,32,1))

keras.utils.plot_model(model_qBOLD_reshaped,"Model_qBOLD_reshaped.png", show_shapes=True)

#%%
layer_idx = layer_names.index("Input_QSM")
QSM_in = model_params.layers[layer_idx]
layer_idx = layer_names.index("spatial_dropout2d_2")
QSM_out = model_params.layers[layer_idx]
layer_idx = layer_names.index("concat_QQ_1")
Concat_in = model_params.layers[layer_idx]
layer_idx = layer_names.index("S0")
S0_out = model_params.layers[layer_idx]
layer_idx = layer_names.index("R2")
R2_out = model_params.layers[layer_idx]
layer_idx = layer_names.index("Y")
Y_out = model_params.layers[layer_idx]
layer_idx = layer_names.index("nu")
nu_out = model_params.layers[layer_idx]
layer_idx = layer_names.index("chi_nb")
chi_out = model_params.layers[layer_idx]

"""



