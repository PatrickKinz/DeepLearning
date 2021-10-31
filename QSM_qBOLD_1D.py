# %% import modules
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
from numpy.random import rand, randn,shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm  #for progress bar
import levenberg_marquardt as lm
import h5py


import time
#%%

#Dataset_train=np.load("../Brain_Phantom/Patches_no_air_big/15GB_1Pnoise_train_val.npz")
#Dataset_test=np.load("../Brain_Phantom/Patches_no_air_big/15GB_1Pnoise_test.npz")
Dataset=np.load("../Brain_Phantom/Patches_no_air/NumpyArchiv.npz")
#S0_train=Dataset_train['S0']
#R2_train=Dataset_train['R2']
#Y_train=Dataset_train['Y']
nu_train=Dataset['Params_training'][:,:,:,3]
nu_train.shape
nu_train=nu_train.reshape(-1,1)
nu_train.shape

#chi_nb_train=Dataset_train['chi_nb']

#S0_test=Dataset_test['S0']
#R2_test=Dataset_test['R2']
#Y_test=Dataset_test['Y']
#nu_test=Dataset_test['nu']
nu_test=Dataset['Params_test'][:,:,:,3]
nu_test=nu_test.reshape(-1,1)
#chi_nb_test=Dataset_test['chi_nb']
"""
qBOLD_training=Dataset_train['qBOLD']
qBOLD_training = qBOLD_training/np.expand_dims(qBOLD_training[:,:,:,0],axis=-1)
t=np.array([3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48])/1000
for i in tqdm(range(qBOLD_training.shape[0])):
    for x in range(qBOLD_training.shape[1]):
        for y in range(qBOLD_training.shape[2]):
            #for ti in range(qBOLD_training.shape[3]):
            qBOLD_training[i,x,y,:]=qBOLD_training[i,x,y,:]/(S0_train[i,x,y]*np.exp(-R2_train[i,x,y] * t))
#qBOLD_test=Dataset_test['qBOLD']
#qBOLD_test = qBOLD_test/np.expand_dims(qBOLD_test[:,:,:,0],axis=-1)
for i in tqdm(range(qBOLD_test.shape[0])):
    for x in range(qBOLD_test.shape[1]):
        for y in range(qBOLD_test.shape[2]):
            #for ti in range(qBOLD_test.shape[3]):
            qBOLD_test[i,x,y,:]=qBOLD_test[i,x,y,:]/(S0_test[i,x,y]*np.exp(-R2_test[i,x,y] * t))
qBOLD_test[0,0,0,:]
#np.savez("../Brain_Phantom/Patches_no_air_big/qBOLD_15GB_S0_R2_removed",qBOLD_training=qBOLD_training,qBOLD_test=qBOLD_test)
"""
#Dataset_reduced=np.load("../Brain_Phantom/Patches_no_air_big/qBOLD_15GB_S0_R2_removed.npz")
qBOLD_train=Dataset['qBOLD_training']
#qBOLD_train=qBOLD_train.reshape(-1,16)
qBOLD_test=Dataset['qBOLD_test']
#qBOLD_test=qBOLD_test.reshape(-1,16)

QSM_train=Dataset['QSM_training']
#QSM_train=QSM_train.reshape(-1,1)
QSM_test=Dataset['QSM_test']
#QSM_test=QSM_test.reshape(-1,1)

features_train =np.concatenate((qBOLD_train,QSM_train), axis=-1)
features_train=features_train.reshape(-1,17)
features_train.shape
features_test =np.concatenate((qBOLD_test,QSM_test), axis=-1)
features_test.shape
features_test=features_test.reshape(-1,17)



#%%
#features_dataset=tf.data.Dataset.from_tensor_slices(features_train)
#labels_dataset=tf.data.Dataset.from_tensor_slices(nu_train)
#train_dataset=tf.data.Dataset.zip((features_dataset,labels_dataset))
train_dataset=tf.data.Dataset.from_tensor_slices((features_train,nu_train))
#batch_size=100000
batch_size=1000
train_dataset = train_dataset.batch(batch_size).cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

#%%
input_qBOLD = keras.Input(shape=(16), name = 'Input_qBOLD')
input_QSM = keras.Input(shape=(1), name = 'Input_QSM')
concat_QQ_1 = layers.Concatenate(name = 'concat_QQ_1')([input_qBOLD,input_QSM]) #model_qBOLD.output,model_QSM.output,

input_features = keras.Input(shape=(17), name = 'Input_features')
Dense_1 = tf.keras.layers.Dense(10, activation='tanh')(input_features)
nu_layer=tf.keras.layers.Dense(1, activation='linear')(Dense_1)
model = keras.Model(inputs=[input_features],outputs=[nu_layer],name="Params_model")

model.summary()
keras.utils.plot_model(model, show_shapes=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError())

model_wrapper = lm.ModelWrapper(
    tf.keras.models.clone_model(model))

model_wrapper.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
    loss=lm.MeanSquaredError(),
    #experimental_use_pfor=False
    )


#%%
print("Train using Adam")
t1_start = time.perf_counter()
model.fit(train_dataset, epochs=1000)
t1_stop = time.perf_counter()
print("Elapsed time: ", t1_stop - t1_start)

#%%
print("\n_________________________________________________________________")
print("Train using Levenberg-Marquardt")
t2_start = time.perf_counter()
model_wrapper.fit(train_dataset, epochs=100)
t2_stop = time.perf_counter()
print("Elapsed time: ", t2_stop - t2_start)
