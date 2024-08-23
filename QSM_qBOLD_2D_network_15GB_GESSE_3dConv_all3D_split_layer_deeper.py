

# %% import modules
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
from tensorflow.keras import layers
#tf.debugging.enable_check_numerics() incredibly slow since check runs on cpu

import os
#os.environ["XLA_FLAGS"]="--xla_gpu_strict_conv_algorithm_picker=false"
#os.environ["XLA_FLAGS"]="--xla_gpu_autotune_level=0"
import json

import numpy as np
from numpy.random import rand, randn,shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm  #for progress bar
import QSM_qBOLD_2D_plotting_functions as QQplt
import QSM_and_qBOLD_functions as QQfunc
import h5py

import MultiInputNumpyArrayGenerator as myGenerator

#from QSM_qBOLD_2D_load_and_prepare_data import load_and_prepare_data


#policy = tf.keras.mixed_precision.Policy('mixed_float16')
#tf.keras.mixed_precision.set_global_policy(policy)
#accelerates training, expecially with tensor cores on RTX cards

#from My_Custom_Generator import My_Params_Generator,My_Signal_Generator
#%%
#data_dir = "../Brain_Phantom/Patches/"
#Params_training,Params_test,qBOLD_training,qBOLD_test,QSM_training,QSM_test = load_and_prepare_data(data_dir)

#np.savez("../Brain_Phantom/Patches/NumpyArchiv",Params_training=Params_training,Params_test=Params_test,qBOLD_training=qBOLD_training,qBOLD_test=qBOLD_test,QSM_training=QSM_training,QSM_test=QSM_test)

#Dataset_train=np.load("../Brain_Phantom/Patches_no_air_big_GESSE/6GB_1Pnoise_train_val_new_tf.npz")
#Dataset_train=np.load("../Brain_Phantom/Patches_no_air_big_GESSE/6GB_1Pnoise_train_val_new_tf.npz")
#size_limit = 30000
#S0_train=tf.data.Dataset.from_tensor_slices(Dataset_train['S0'][0:size_limit,:,:,:])
#R2_train=tf.data.Dataset.from_tensor_slices(Dataset_train['R2'][0:size_limit,:,:,:])
#Y_train=tf.data.Dataset.from_tensor_slices(Dataset_train['Y'][0:size_limit,:,:,:])
#nu_train=tf.data.Dataset.from_tensor_slices(Dataset_train['nu'][0:size_limit,:,:,:])
#chi_nb_train=tf.data.Dataset.from_tensor_slices(Dataset_train['chi_nb'][0:size_limit,:,:,:])
#qBOLD_training=tf.data.Dataset.from_tensor_slices(Dataset_train['qBOLD'][0:size_limit,:,:,:])
#QSM_training=tf.data.Dataset.from_tensor_slices(Dataset_train['QSM'][0:size_limit,:,:,:])

#10% validation = 42441*0.9 = 38197
#1% validation = 42441*0.99 = 42017 = 42000
#val_border = 27000
#training_list = [S0_train[0:val_border,:,:,:],R2_train[0:val_border,:,:,:],Y_train[0:val_border,:,:,:],nu_train[0:val_border,:,:,:],chi_nb_train[0:val_border,:,:,:]]
#validation_list =[S0_train[val_border:-1,:,:,:],R2_train[val_border:-1,:,:,:],Y_train[val_border:-1,:,:,:],nu_train[val_border:-1,:,:,:],chi_nb_train[val_border:-1,:,:,:]]
#input_train_list = [qBOLD_training[0:val_border,:,:,:],QSM_training[0:val_border,:,:,:]]
#input_val_list = [qBOLD_training[val_border:-1,:,:,:],QSM_training[val_border:-1,:,:,:]]

#%%
# List of .npz files
directory = "../Brain_Phantom/Patches_no_air_big_GESSE_1pNoise_single_files_train_val_1_input/"
npz_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]


# Shuffle the list of files
np.random.shuffle(npz_files)

# Define the split ratio
split_ratio = 0.8  # 80% training, 20% validation

# Calculate the split index
split_index = int(len(npz_files) * split_ratio)

# Split the files into training and validation sets
train_files = npz_files[:split_index]
val_files = npz_files[split_index:]

# Parameters
batch_size = 100
input_keys = ['qqBOLD']  # Replace with your actual input keys
label_keys = ['S0', 'R2','Y','nu','chi_nb']  # Replace with your actual label keys




# Instantiate the generator
train_generator = myGenerator.MultiInputNumpyArrayGenerator(train_files, batch_size, input_keys, label_keys)
val_generator = myGenerator.MultiInputNumpyArrayGenerator(val_files, batch_size, input_keys, label_keys)

# Example of using with a model
# model.fit(train_generator, epochs=10)
#%%
# List of .npz files
directory_test = "../Brain_Phantom/Patches_no_air_big_GESSE_1pNoise_single_files_test_1_input/"
test_files = [os.path.join(directory_test, f) for f in os.listdir(directory_test) if f.endswith('.npz')]

# Parameters
batch_size = 100
input_keys = ['qqBOLD']  # Replace with your actual input keys
label_keys = ['S0', 'R2','Y','nu','chi_nb']  # Replace with your actual label keys

# Instantiate the generator
test_generator = myGenerator.MultiInputNumpyArrayGenerator(test_files, batch_size, input_keys, label_keys)



#%%
version = "no_air_1Pnoise_15GB_GESSE/"
model_folder_name = 'model_GESSE_n16_all3D_deeper_1P/'

# Function to split the tensor into two parts
@tf.keras.utils.register_keras_serializable()
def split_tensor(x):
    output1 = x[:,:,:, :-1,:]
    output2 = tf.expand_dims(x[:,:,:, -1,:], axis=-1)
    return output1, output2



# %% Network
input_qqBOLD = keras.Input(shape=(None,None,33,1),name='qqBOLD')


# Lambda layer to apply the split
split1, split2 = keras.layers.Lambda(split_tensor)(input_qqBOLD)

#input_qBOLD = keras.Input(shape=(None,None,32,1), name = 'qBOLD')



n=16
pad_qBOLD_1 = keras.layers.ZeroPadding3D(padding=(1,1,0))(split1)
conv_qBOLD_1 = keras.layers.Conv3D(n,
                  kernel_size = (3,3,7),
                  strides=1,
                  padding='valid',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_qBOLD_1')(pad_qBOLD_1)
norm_qBOLD_1 = layers.BatchNormalization()(conv_qBOLD_1)
drop_qBOLD_1 = layers.SpatialDropout3D(0.1)(norm_qBOLD_1)

pad_qBOLD_2 = keras.layers.ZeroPadding3D(padding=(1,1,0))(drop_qBOLD_1)
conv_qBOLD_2 = keras.layers.Conv3D(2*n,
                  kernel_size = (3,3,7),
                  strides=1,
                  padding='valid',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_qBOLD_2')(pad_qBOLD_2)
norm_qBOLD_2 = layers.BatchNormalization()(conv_qBOLD_2)
drop_qBOLD_2 = layers.SpatialDropout3D(0.1)(norm_qBOLD_2)

pad_qBOLD_3 = keras.layers.ZeroPadding3D(padding=(1,1,0))(conv_qBOLD_2)
conv_qBOLD_3 = keras.layers.Conv3D(4*n,
                  kernel_size = (3,3,7),
                  strides=1,
                  padding='valid',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_qBOLD_3')(pad_qBOLD_3)
norm_qBOLD_3 = layers.BatchNormalization()(conv_qBOLD_3)
drop_qBOLD_3 = layers.SpatialDropout3D(0.1)(norm_qBOLD_3)

pad_qBOLD_4 = keras.layers.ZeroPadding3D(padding=(1,1,0))(conv_qBOLD_3)
conv_qBOLD_4 = keras.layers.Conv3D(8*n,
                  kernel_size = (3,3,7),
                  strides=1,
                  padding='valid',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_qBOLD_4')(pad_qBOLD_4)
norm_qBOLD_4 = layers.BatchNormalization()(conv_qBOLD_4)
drop_qBOLD_4 = layers.SpatialDropout3D(0.1)(norm_qBOLD_4)

pad_qBOLD_5 = keras.layers.ZeroPadding3D(padding=(1,1,0))(conv_qBOLD_4)
conv_qBOLD_5 = keras.layers.Conv3D(16*n,
                   kernel_size = (3,3,8),
                   strides=1,
                   padding='valid',
                   dilation_rate=1,
                   activation='tanh',
                   name='conv_qBOLD_5')(pad_qBOLD_5)
norm_qBOLD_5 = layers.BatchNormalization()(conv_qBOLD_5)
drop_qBOLD_5 = layers.SpatialDropout3D(0.1)(norm_qBOLD_5)

#pad_qBOLD_6 = keras.layers.ZeroPadding3D(padding=(1,1,0))(conv_qBOLD_5)
#conv_qBOLD_6 = keras.layers.Conv3D(32*n,
#                   kernel_size = (3,3,7),
#                   strides=1,
#                   padding='valid',
#                   dilation_rate=1,
#                   activation='tanh',
#                   name='conv_qBOLD_6')(pad_qBOLD_6)
#norm_qBOLD_6 = layers.BatchNormalization()(conv_qBOLD_6)
#drop_qBOLD_6 = layers.SpatialDropout3D(0.1)(norm_qBOLD_6)

#newdim = tuple([x for x in drop_qBOLD_4.shape.as_list() if x != 1 and x is not None])
#reshape_qBOLD = keras.layers.Reshape(newdim) (drop_qBOLD_4) #should remove dimensions of size 1
#reshape_qBOLD = keras.layers.Reshape((30,30,128))(drop_qBOLD_4)

#model_qBOLD = keras.Model(inputs=input_qqBOLD, outputs = drop_qBOLD_5, name="qBOLD model")
#model_qBOLD.summary()

#keras.utils.plot_model(model_qBOLD, show_shapes=True)

#input_QSM = keras.Input(shape=(None,None,1,1), name = 'QSM')
conv_QSM_1 = keras.layers.Conv3D(int(n/2),
                  kernel_size=3,
                  strides=(1),
                  padding='same',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_QSM_1')(split2)
norm_QSM_1 = layers.BatchNormalization()(conv_QSM_1)
drop_QSM_1 = layers.SpatialDropout3D(0.1)(norm_QSM_1)

conv_QSM_2 = keras.layers.Conv3D(n,
                  kernel_size=3,
                  strides=(1),
                  padding='same',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_QSM_2')(drop_QSM_1)
norm_QSM_2 = layers.BatchNormalization()(conv_QSM_2)
drop_QSM_2 = layers.SpatialDropout3D(0.1)(norm_QSM_2)

conv_QSM_3 = keras.layers.Conv3D(2*n,
                  kernel_size=3,
                  strides=(1),
                  padding='same',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_QSM_3')(drop_QSM_2)
norm_QSM_3 = layers.BatchNormalization()(conv_QSM_3)
drop_QSM_3 = layers.SpatialDropout3D(0.1)(norm_QSM_3)



#model_QSM = keras.Model(inputs=input_QSM, outputs = drop_QSM_3, name="QSM model")
#model_QSM.summary()
#keras.utils.plot_model(model_QSM, show_shapes=True)

concat_QQ_1 = layers.Concatenate(name = 'concat_QQ_1')([drop_qBOLD_5,drop_QSM_3])
conv_QQ_1 = layers.Conv3D(8*n,3,padding='same',activation="tanh",name = 'conv_QQ_1')(concat_QQ_1)
#conv_QQ_1 = layers.Conv3D(2*n,3,padding='same',activation="tanh",name = 'conv_QQ_1')(drop_qBOLD_4)
norm_QQ_1 = layers.BatchNormalization()(conv_QQ_1)
drop_QQ_1 = layers.SpatialDropout3D(0.1)(norm_QQ_1)

conv_QQ_2 = layers.Conv3D(16*n,3,padding='same',activation="tanh",name = 'conv_QQ_2')(drop_QQ_1)
norm_QQ_2 = layers.BatchNormalization()(conv_QQ_2)
drop_QQ_2 = layers.SpatialDropout3D(0.1)(norm_QQ_2)

conv_QQ_3 = layers.Conv3D(32*n,3,padding='same',activation="tanh",name = 'conv_QQ_3')(drop_QQ_2)
norm_QQ_3 = layers.BatchNormalization()(conv_QQ_3)
drop_QQ_3 = layers.SpatialDropout3D(0.1)(norm_QQ_3)

conv_S0_1 = layers.Conv3D(32*n,3,padding='same',activation="tanh",name = 'conv_S0_1')(drop_QQ_3)
norm_S0_1 = layers.BatchNormalization()(conv_S0_1)
drop_S0_1 = layers.SpatialDropout3D(0.1)(norm_S0_1)

conv_R2_1 = layers.Conv3D(32*n,3,padding='same',activation="tanh",name = 'conv_R2_1')(drop_QQ_3)
norm_R2_1 = layers.BatchNormalization()(conv_R2_1)
drop_R2_1 = layers.SpatialDropout3D(0.1)(norm_R2_1)

conv_Y_1 = layers.Conv3D(32*n,3,padding='same',activation="tanh",name = 'conv_Y_1')(drop_QQ_3)
norm_Y_1 = layers.BatchNormalization()(conv_Y_1)
drop_Y_1 = layers.SpatialDropout3D(0.1)(norm_Y_1)

conv_nu_1 = layers.Conv3D(32*n,3,padding='same',activation="tanh",name = 'conv_nu_1')(drop_QQ_3)
norm_nu_1 = layers.BatchNormalization()(conv_nu_1)
drop_nu_1 = layers.SpatialDropout3D(0.1)(norm_nu_1)

conv_chinb_1 = layers.Conv3D(32*n,3,padding='same',activation="tanh",name = 'conv_chinb_1')(drop_QQ_3)
norm_chinb_1 = layers.BatchNormalization()(conv_chinb_1)
drop_chinb_1 = layers.SpatialDropout3D(0.1)(norm_chinb_1)


conv_S0 = layers.Conv3D(1,3,padding='same',activation="linear", name = 'S0')(    drop_S0_1)
conv_R2 = layers.Conv3D(1,3,padding='same',activation="linear", name = 'R2')(    drop_R2_1)
conv_Y = layers.Conv3D(1,3,padding='same',activation="linear", name = 'Y')(     drop_Y_1)
conv_nu = layers.Conv3D(1,3,padding='same',activation="linear", name = 'nu')(    drop_nu_1)
conv_chinb = layers.Conv3D(1,3,padding='same',activation="linear", name = 'chi_nb')(drop_chinb_1)


model_params = keras.Model(inputs=input_qqBOLD,outputs=[conv_S0,conv_R2,conv_Y,conv_nu,conv_chinb],name="Params_model")
model_params.summary()
#keras.utils.plot_model(model_params, show_shapes=True)




# %% Train Params model

opt = keras.optimizers.Adam(0.001, clipnorm=1.)
#loss=keras.losses.MeanAbsolutePercentageError()
#loss=keras.losses.MeanSquaredLogarithmicError()
#loss=keras.losses.MeanAbsoluteError()
#loss=tf.keras.losses.Huber()
losses = {
    "S0":keras.losses.MeanAbsoluteError(name='loss_S0'),
    "R2":keras.losses.MeanAbsoluteError(name='loss_R2'),
    "Y":keras.losses.MeanAbsoluteError(name='loss_Y'),
    "nu":keras.losses.MeanAbsoluteError(name='loss_nu'),
    "chi_nb":keras.losses.MeanAbsoluteError(name='loss_chi_nb'),
}
lossWeights = {
    "S0":1.0,
    "R2":1.0,
    "Y":1.0,
    "nu":1.0,
    "chi_nb":1.0,
}
metrics = {
    "S0":keras.metrics.MeanAbsoluteError(name='metric_S0'),
    "R2":keras.metrics.MeanAbsoluteError(name='metric_R2'),
    "Y":keras.metrics.MeanAbsoluteError(name='metric_Y'),
    "nu":keras.metrics.MeanAbsoluteError(name='metric_nu'),
    "chi_nb":keras.metrics.MeanAbsoluteError(name='metric_chi_nb'),
}
model_params.compile(
    loss=losses,
    loss_weights=lossWeights,
    optimizer=opt,
    metrics=metrics,
    #metrics=[tf.keras.metrics.MeanSquaredError()],
    #metrics=["accuracy"],
)

# Add multiple metrics to get a loss for each output parameter, make sure they are saved too

#%%
my_callbacks = [
    #tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath='models/' + model_folder_name +'checkpoints/model.{epoch:02d}-{val_loss:.4f}.keras'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]


#initial_epoch = 5 would start training at epoch 6
# model.fit(train_generator, epochs=10)
history_params = model_params.fit(train_generator , epochs=50, validation_data=val_generator, callbacks=my_callbacks)
#history_params = model_params.fit([qBOLD_training,QSM_training], training_list , batch_size=5, epochs=20, validation_split=0.1, callbacks=my_callbacks)
#history_params = model_params.fit(training_Params_data, epochs=100,validation_data=val_Params_data, callbacks=my_callbacks)
#%%
model_params.save('models/' + model_folder_name + "Model_2D_image_GESSE_3D_conv_norm_drop_n16_all3D_deeper.keras")
np.save('models/' + model_folder_name +'history_Model_2D_image_GESSE_3D_conv_norm_drop_n16_all3D_deeper.npy',history_params.history)
with open('models/' + model_folder_name +'history_Model_2D_image_GESSE_3D_conv_norm_drop_n16_all3D_deeper.json', 'w') as f:
    json.dump(history_params.history, f)
#%%

model_params = keras.models.load_model('models/' + model_folder_name + "Model_2D_image_GESSE_3D_conv_norm_drop_n16_all3D_newTF.keras")
#model_params = keras.models.load_model("checkpoints/model.32-0.31.keras")
model_params.summary()
keras.utils.plot_model(model_params, show_shapes=True)


#%%


test_scores = model_params.evaluate(test_generator, verbose=1,return_dict=True)
print(test_scores)
with open('models/' + model_folder_name +'Model_2D_image_GESSE_3D_conv_norm_drop_n16_all3D_deeper_test_scores.json', 'w') as f:
    json.dump(test_scores, f)

#print("Test accuracy:", test_scores[1])
#%%
history_params = np.load(version+'history_Model_2D_image_GESSE_3D_conv_norm_drop_n16_all3D_newTF2.npy',allow_pickle=True)
#%%
print(history_params.history.keys())

QQplt.plot_loss(history_params,'')
QQplt.plot_loss(history_params,'S0_')
QQplt.plot_loss(history_params,'R2_')
QQplt.plot_loss(history_params,'Y_')
QQplt.plot_loss(history_params,'nu_')
QQplt.plot_loss(history_params,'chi_nb_')


# %%
#model_params = keras.models.load_model("models/"+version+ "Model_2D_Params_before_qqbold.h5")
#model_params.summary()


#%%
#how do I load all the data?


def combine_npz_arrays(folder_path):
  """Combines arrays from multiple npz files into a single dictionary.

  Args:
    folder_path: Path to the folder containing .npz files.

  Returns:
    A dictionary where keys are the same as in the npz files and values are concatenated arrays.
  """

  combined_data = {}
  for file in os.listdir(folder_path):
    if file.endswith('.npz'):
      file_path = os.path.join(folder_path, file)
      npz_data = np.load(file_path)
      for key, array in npz_data.items():
        if key not in combined_data:
          combined_data[key] = [array]
        else:
          combined_data[key].append(array)
      
  # Convert lists of arrays to numpy arrays
  for key, array_list in combined_data.items():
    combined_data[key] = np.stack(array_list, axis=0)  # Adjust axis if needed

  return combined_data



combined_data = combine_npz_arrays(directory_test)
print(combined_data)

#%%
label_transformed=QQplt.translate_Params([np.array(combined_data['S0']),np.array(combined_data['R2']),np.array(combined_data['Y']),np.array(combined_data['nu']),np.array(combined_data['chi_nb'])])

#%%
p = model_params.predict(combined_data['qqBOLD'])
p[0].shape

prediction_transformed=QQplt.translate_Params(p)



#%%
Number=25
QQplt.check_Params_transformed(label_transformed,prediction_transformed,Number,'CNN_Uniform_GESSE_1Pnoise_32Echoes_Params_3d_drop_norm_n16_all3D_newTF')
#%%
QQplt.check_Params_transformed_hist(label_transformed,prediction_transformed,'CNN_Uniform_GESSE_1Pnoise_32Echoes_evaluation_3D_drop_norm_n16_all3D_newTF')
# this created the ISMRM 2022 plot for Gesfide
# add full histogram plot here
#%%
print(label_transformed[0].shape)
print(prediction_transformed[0].shape)
for i in range(len(prediction_transformed)):
    label_transformed[i] = label_transformed[i].flatten()
    prediction_transformed[i] = prediction_transformed[i].flatten()
print(prediction_transformed[0].shape)
print(label_transformed[0].shape)
# add full histogram plot here
#QQplt.check_full_confusion_matrix(label_transformed,prediction_transformed,'confusion_test_GESSE_1Pnoise_3d_drop_norm_n16')
#QQplt.check_full_confusion_matrix_autonormed(label_transformed,prediction_transformed,'confusion_test_GESSE_1Pnoise_3d_drop_norm_n16_autonormed')
#%%
QQplt.check_full_confusion_matrix_normed(label_transformed,prediction_transformed,'confusion_test_GESSE_1Pnoise_3d_drop_norm_n16_all3D_splitinput_deeper')

#QQplt.check_correlation_coef(label_transformed,prediction_transformed,'confusion_test_GESSE_1Pnoise_correlation_coeffs_3d_drop_norm_n16')
#%%
QQplt.check_full_confusion_matrix_normed(prediction_transformed,prediction_transformed,'confusion_test_GESSE_1Pnoise_3d_drop_norm_n16_all3D_splitinput_deeper_pvp')


# %%
