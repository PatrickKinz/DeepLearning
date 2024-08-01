# %% import modules
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
from tensorflow.keras import layers

#tf.debugging.enable_check_numerics() incredibly slow since check runs on cpu


import numpy as np
from numpy.random import rand, randn,shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm  #for progress bar
import QSM_qBOLD_2D_plotting_functions as QQplt
import QSM_and_qBOLD_functions as QQfunc
import h5py
#from QSM_qBOLD_2D_load_and_prepare_data import load_and_prepare_data


#policy = tf.keras.mixed_precision.Policy('mixed_float16')
#tf.keras.mixed_precision.set_global_policy(policy)
#accelerates training, expecially with tensor cores on RTX cards

#from My_Custom_Generator import My_Params_Generator,My_Signal_Generator
#%%
#data_dir = "../Brain_Phantom/Patches/"
#Params_training,Params_test,qBOLD_training,qBOLD_test,QSM_training,QSM_test = load_and_prepare_data(data_dir)

#np.savez("../Brain_Phantom/Patches/NumpyArchiv",Params_training=Params_training,Params_test=Params_test,qBOLD_training=qBOLD_training,qBOLD_test=qBOLD_test,QSM_training=QSM_training,QSM_test=QSM_test)

Dataset_train=np.load("../Brain_Phantom/Patches_no_air_big_triple_GRE/15GB_1Pnoise_train_val.npz")
S0_train=Dataset_train['S0']
S0_train.shape
R2_train=Dataset_train['R2']
Y_train=Dataset_train['Y']
nu_train=Dataset_train['nu']
chi_nb_train=Dataset_train['chi_nb']
T1_train=Dataset_train['T1']
#qBOLD_training=Dataset_train['qBOLD'][:,:,:,0:5]
qBOLD_training=Dataset_train['qBOLD']
QSM_training=Dataset_train['QSM']


training_list = [S0_train,R2_train,Y_train,nu_train,chi_nb_train,T1_train]
#%%
Dataset_test=np.load("../Brain_Phantom/Patches_no_air_big_triple_GRE/15GB_1Pnoise_test.npz")
S0_test=Dataset_test['S0']
S0_test.shape
R2_test=Dataset_test['R2']
Y_test=Dataset_test['Y']
nu_test=Dataset_test['nu']
chi_nb_test=Dataset_test['chi_nb']
T1_test=Dataset_test['T1']
#qBOLD_test=Dataset_test['qBOLD'][:,:,:,0:5]
qBOLD_test=Dataset_test['qBOLD']
QSM_test=Dataset_test['QSM']

test_list = [S0_test,R2_test,Y_test,nu_test,chi_nb_test,T1_test]


version = "no_air_1Pnoise_15GB_triple_GRE/"

# %% Network

input_qBOLD = keras.Input(shape=(30,30,24,1), name = 'Input_qBOLD')

input_qBOLD.shape
input_qBOLD.dtype

n=16
pad_qBOLD_1 = keras.layers.ZeroPadding3D(padding=(1,1,0))(input_qBOLD)
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

pad_qBOLD_3 = keras.layers.ZeroPadding3D(padding=(1,1,0))(drop_qBOLD_2)
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
                  kernel_size = (3,3,6),
                  strides=1,
                  padding='valid',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_qBOLD_4')(pad_qBOLD_4)
norm_qBOLD_4 = layers.BatchNormalization()(conv_qBOLD_4)
drop_qBOLD_4 = layers.SpatialDropout3D(0.1)(norm_qBOLD_4)

# pad_qBOLD_5 = keras.layers.ZeroPadding3D(padding=(1,1,0))(conv_qBOLD_4)
# conv_qBOLD_5 = keras.layers.Conv3D(16*n,
#                   kernel_size = (3,3,9),
#                   strides=1,
#                   padding='valid',
#                   dilation_rate=1,
#                   activation='tanh',
#                   name='conv_qBOLD_5')(pad_qBOLD_5)
# norm_qBOLD_5 = layers.BatchNormalization()(conv_qBOLD_5)
# drop_qBOLD_5 = layers.SpatialDropout3D(0.1)(norm_qBOLD_5)

# pad_qBOLD_6 = keras.layers.ZeroPadding3D(padding=(1,1,0))(conv_qBOLD_5)
# conv_qBOLD_6 = keras.layers.Conv3D(32*n,
#                   kernel_size = (3,3,4),
#                   strides=1,
#                   padding='valid',
#                   dilation_rate=1,
#                   activation='tanh',
#                   name='conv_qBOLD_6')(pad_qBOLD_6)
# norm_qBOLD_6 = layers.BatchNormalization()(conv_qBOLD_6)
# drop_qBOLD_6 = layers.SpatialDropout3D(0.1)(norm_qBOLD_6)


reshape_qBOLD = keras.layers.Reshape((30,30,128))(drop_qBOLD_4)

model_qBOLD = keras.Model(inputs=input_qBOLD, outputs = reshape_qBOLD, name="qBOLD model")
model_qBOLD.summary()
keras.utils.plot_model(model_qBOLD, show_shapes=True)

#%%
input_QSM = keras.Input(shape=(30,30,3), name = 'Input_QSM')
conv_QSM_1 = keras.layers.Conv2D(n/2,
                  kernel_size=3,
                  strides=(1),
                  padding='same',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_QSM_1')(input_QSM)
norm_QSM_1 = layers.BatchNormalization()(conv_QSM_1)
drop_QSM_1 = layers.SpatialDropout2D(0.1)(norm_QSM_1)

conv_QSM_2 = keras.layers.Conv2D(n,
                  kernel_size=3,
                  strides=(1),
                  padding='same',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_QSM_2')(drop_QSM_1)
norm_QSM_2 = layers.BatchNormalization()(conv_QSM_2)
drop_QSM_2 = layers.SpatialDropout2D(0.1)(norm_QSM_2)

conv_QSM_3 = keras.layers.Conv2D(2*n,
                  kernel_size=3,
                  strides=(1),
                  padding='same',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_QSM_3')(drop_QSM_2)
norm_QSM_3 = layers.BatchNormalization()(conv_QSM_3)
drop_QSM_3 = layers.SpatialDropout2D(0.1)(norm_QSM_3)



model_QSM = keras.Model(inputs=input_QSM, outputs = drop_QSM_3, name="QSM model")
model_QSM.summary()
keras.utils.plot_model(model_QSM, show_shapes=True)
#%%
concat_QQ_1 = layers.Concatenate(name = 'concat_QQ_1')([model_qBOLD.output,model_QSM.output])
conv_QQ_1 = layers.Conv2D(2*n,3,padding='same',activation="tanh",name = 'conv_QQ_1')(concat_QQ_1)
norm_QQ_1 = layers.BatchNormalization()(conv_QQ_1)
drop_QQ_1 = layers.SpatialDropout2D(0.1)(norm_QQ_1)

conv_QQ_2 = layers.Conv2D(4*n,3,padding='same',activation="tanh",name = 'conv_QQ_2')(drop_QQ_1)
norm_QQ_2 = layers.BatchNormalization()(conv_QQ_2)
drop_QQ_2 = layers.SpatialDropout2D(0.1)(norm_QQ_2)

conv_QQ_3 = layers.Conv2D(8*n,3,padding='same',activation="tanh",name = 'conv_QQ_3')(drop_QQ_2)
norm_QQ_3 = layers.BatchNormalization()(conv_QQ_3)
drop_QQ_3 = layers.SpatialDropout2D(0.1)(norm_QQ_3)



conv_S0 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'S0')(    drop_QQ_3)
conv_R2 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'R2')(    drop_QQ_3)
conv_Y = layers.Conv2D(1,3,padding='same',activation="linear", name = 'Y')(     drop_QQ_3)
conv_nu = layers.Conv2D(1,3,padding='same',activation="linear", name = 'nu')(    drop_QQ_3)
conv_chinb = layers.Conv2D(1,3,padding='same',activation="linear", name = 'chi_nb')(drop_QQ_3)
conv_T1 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'T1')(    drop_QQ_3)


model_params = keras.Model(inputs=[input_qBOLD,input_QSM],outputs=[conv_S0,conv_R2,conv_Y,conv_nu,conv_chinb,conv_T1],name="Params_model")
model_params.summary()
keras.utils.plot_model(model_params, show_shapes=True)




# %% Train Params model

opt = keras.optimizers.Adam(0.001, clipnorm=1.)
#loss=keras.losses.MeanAbsolutePercentageError()
#loss=keras.losses.MeanSquaredLogarithmicError()
loss=keras.losses.MeanAbsoluteError()
#loss=tf.keras.losses.Huber()
losses = {
    "S0":loss,
    "R2":loss,
    "Y":loss,
    "nu":loss,
    "chi_nb":loss,
    "T1":loss,
}
lossWeights = {
    "S0":1.0,
    "R2":1.0,
    "Y":1.0,
    "nu":1.0,
    "chi_nb":1.0,
    "T1":1.0,
}
model_params.compile(
    loss=losses,
    loss_weights=lossWeights,
    optimizer=opt,
    #metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
    #metrics=[tf.keras.metrics.MeanSquaredError()],
    #metrics=["accuracy"],
)

#%%
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]




history_params = model_params.fit([qBOLD_training,QSM_training], training_list , batch_size=10, epochs=100, validation_split=0.1/0.9, callbacks=my_callbacks)
#history_params = model_params.fit(training_Params_data, epochs=100,validation_data=val_Params_data, callbacks=my_callbacks)
#%%
#model_params.save("models/"+version+ "Model_2D_image_triple_GRE_1Pnoise_3D_conv_norm_drop.h5")
#np.save('models/'+version+'history_Model_2D_image_triple_GRE_1Pnoise_3D_conv_norm_drop.npy',history_params.history)
model_params = keras.models.load_model("models/"+version+ "Model_2D_image_triple_GRE_1Pnoise_3D_conv_norm_drop.h5")
#model_params.summary()
#keras.utils.plot_model(model_params, show_shapes=True)

#%%


test_scores = model_params.evaluate([qBOLD_test,QSM_test], test_list, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
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
p = model_params.predict([qBOLD_test,QSM_test])
p[0].shape

#%%
Number=2
label_transformed=QQplt.translate_Params(test_list)
prediction_transformed=QQplt.translate_Params(p)
QQplt.check_Params_transformed(label_transformed,prediction_transformed,Number,'CNN_triple_GRE_1Pnoise_Params_3d_drop_norm_n16')

QQplt.check_Params_transformed_hist(label_transformed,prediction_transformed,'CNN_triple_GRE_1Pnoise_evaluation_3D_drop_norm_n16')
# this created the ISMRM 2022 plot for Gesfide
# add full histogram plot here
label_transformed[0].shape
prediction_transformed[0].shape
for i in range(len(prediction_transformed)):
    label_transformed[i] = label_transformed[i].flatten()
    prediction_transformed[i] = prediction_transformed[i].flatten()
prediction_transformed[0].shape
label_transformed[0].shape
# add full histogram plot here
QQplt.check_full_confusion_matrix_normed(label_transformed,prediction_transformed,'confusion_test_triple_GRE_1Pnoise_3d_drop_norm_n16')
QQplt.check_correlation_coef(label_transformed,prediction_transformed,'confusion_test_triple_GRE_1Pnoise_correlation_coeffs_3d_drop_norm_n16')


# %%
