#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sp

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
from tensorflow.keras import layers

import QSM_qBOLD_2D_plotting_functions as QQplt
import QSM_and_qBOLD_functions as QQfunc


# %%
mat_contents = sp.loadmat("D:/AG_TSF_local/ClinicalStudies/OEF_Healthy_UMM/20180504_SH/results/Gesse.mat")
print(mat_contents.keys())

mag_gesse = mat_contents.get("mag_gesse")
iField_gesse = mat_contents.get("iField_gesse")
iField_gesse_reg = mat_contents.get("iField_gesse_reg")
mag_gesse_cor = mat_contents.get("mag_gesse_cor")
Mask_gesse = mat_contents.get("Mask_gesse")
# %%
mat_contents = sp.loadmat("D:/AG_TSF_local/ClinicalStudies/OEF_Healthy_UMM/20180504_SH/results/OEF_ANN.mat")
print(mat_contents.keys())
OEF_ANN = mat_contents.get("OEF_ANN")
# %%
mat_contents = sp.loadmat("D:/AG_TSF_local/ClinicalStudies/OEF_Healthy_UMM/20180504_SH/results/QSM.mat")
print(mat_contents.keys())
QSM_gesse = mat_contents.get("QSM_gesse")

# %%
""" mag_gesse and QSM_gesse as input for Neural network"""

n_slice = 11

crop_x=20 #40
crop_y=17# 30
size_x = 90#30
size_y = 60#size_x
plt.figure()
plt.imshow(mag_gesse_cor[crop_x:crop_x+size_x,crop_y:crop_y+size_y,n_slice,0])
plt.axis('off')
plt.show()


plt.figure()
plt.imshow(QSM_gesse[crop_x:crop_x+size_x,crop_y:crop_y+size_y,n_slice])
plt.axis('off')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Mask_gesse[crop_x:crop_x+size_x,crop_y:crop_y+size_y,n_slice])
plt.axis('off')
plt.colorbar()
plt.show()


mag_gesse_rearr = np.moveaxis(mag_gesse_cor,2,0)
QSM_gesse_rearr = np.moveaxis(QSM_gesse,2,0)
Mask_gesse_rearr = np.moveaxis(Mask_gesse,2,0)

#%%
"""slice image into 30 * 30 patches"""
n_pps = 2*3 #number of patches per slice
patch_size = 30
n_echoes = 32
qBOLD_array  = np.zeros((n_pps,patch_size,patch_size,n_echoes),dtype=np.float32)
QSM_array    = np.zeros((n_pps,patch_size,patch_size),dtype=np.float32)
fig, axes = plt.subplots(nrows=3, ncols=2,figsize=(5,7))
ax = axes.ravel()
    
for x in range(3):
    for y in range(2):
        print(x*2+y)
        qBOLD_array[x*2+y,:,:,:]=mag_gesse_rearr[n_slice,crop_x+x*patch_size:crop_x+(x+1)*patch_size,crop_y+y*patch_size:crop_y+(y+1)*patch_size,:]
        QSM_array[x*2+y,:,:]=QSM_gesse_rearr[n_slice,crop_x+x*patch_size:crop_x+(x+1)*patch_size,crop_y+y*patch_size:crop_y+(y+1)*patch_size]
        ax[x*2+y].imshow(QSM_array[x*2+y,:,:])
        ax[x*2+y].get_xaxis().set_visible(False)
        ax[x*2+y].get_yaxis().set_visible(False)
    
plt.show()
qBOLD_array = qBOLD_array/np.max(qBOLD_array)

# %%
version = "no_air_1Pnoise_15GB_GESSE/"

model_params = keras.models.load_model("models/"+version+ "Model_2D_image_GESSE_3D_conv_norm_drop_n16.h5")
model_params.summary()
#%%
keras.utils.plot_model(model_params, show_shapes=True)




#%%
#p = model_params.predict([mag_gesse_rearr[:,crop_x:crop_x+size_x,crop_y:crop_y+size_y,:],QSM_gesse_rearr[:,crop_x:crop_x+size_x,crop_y:crop_y+size_y]])
p = model_params.predict([qBOLD_array,QSM_array])
p[0].shape



#label_transformed=QQplt.translate_Params(test_list)
prediction_transformed=QQplt.translate_Params(p)

#%%
Number=1
QQplt.check_Params_transformed(prediction_transformed,prediction_transformed,Number,'CNN_Uniform_GESSE_1Pnoise_32Echoes_Params_3d_drop_norm_n16_real_data')

#%%

Mask_Gesse_crop=Mask_gesse_rearr[n_slice,crop_x:crop_x+size_x,crop_y:crop_y+size_y]
def plot_prediction(p,mask):
    nx = 3
    ny = 2
    patch_size =30
    image = []
    plots = []
    for i in range(len(p)):
        image.append( np.zeros((nx*patch_size,ny*patch_size)))
        for x in range(nx):
            for y in range(ny):
                image[i][x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size]=p[i][x*ny+y,:,:,0]
    
    fig, axes = plt.subplots(nrows=1, ncols=5,figsize=(10,2))
    ax = axes.ravel()
    P0=ax[0].imshow(image[0]*mask,cmap='gray')
    P0.set_clim(.0,1)
    ax[0].title.set_text('$S_0$ [a.u.]')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    plt.colorbar(P0,ax=ax[0])
    P1=ax[1].imshow(image[1]*mask,cmap='inferno')
    P1.set_clim(0,30)
    ax[1].title.set_text('$R_2$ [Hz]')
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    plt.colorbar(P1,ax=ax[1])
    P2=ax[2].imshow((1-image[2]/0.98)*mask*100,cmap='inferno')
    P2.set_clim(.0,1*100)
    ax[2].title.set_text('OEF [%]')
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    plt.colorbar(P2,ax=ax[2])
    P3=ax[3].imshow(image[3]*mask*100,cmap='inferno')
    P3.set_clim(0,0.1*100)
    ax[3].title.set_text('$v$ [%]')
    ax[3].get_xaxis().set_visible(False)
    ax[3].get_yaxis().set_visible(False)
    plt.colorbar(P3,ax=ax[3])
    P4=ax[4].imshow(image[4]*mask*1000,cmap='gray')
    P4.set_clim(-.1*1000,.1*1000)
    ax[4].title.set_text('$\chi_{nb}$ [ppb]')
    ax[4].get_xaxis().set_visible(False)
    ax[4].get_yaxis().set_visible(False)
    plt.colorbar(P4,ax=ax[4])
    plt.show()
    fig.savefig('plots/real_brain_gesse_slice_11.png')
    
plot_prediction(prediction_transformed,Mask_Gesse_crop) 

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



