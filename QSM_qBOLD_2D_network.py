# %% import modules
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt
from tqdm import tqdm  #for progress bar

import h5py
from QSM_qBOLD_2D_load_and_prepare_data import load_and_prepare_data

#%%
#data_dir = "../Brain_Phantom/Patches/"
#Params_training,Params_test,qBOLD_training,qBOLD_test,QSM_training,QSM_test = load_and_prepare_data(data_dir)

#np.savez("../Brain_Phantom/Patches/NumpyArchiv",Params_training=Params_training,Params_test=Params_test,qBOLD_training=qBOLD_training,qBOLD_test=qBOLD_test,QSM_training=QSM_training,QSM_test=QSM_test)
Dataset=np.load("../Brain_Phantom/Patches/NumpyArchiv.npz")
Params_training=Dataset['Params_training']
Params_test=Dataset['Params_test']
qBOLD_training=Dataset['qBOLD_training']
qBOLD_test=Dataset['qBOLD_test']
QSM_training=Dataset['QSM_training']
QSM_test=Dataset['QSM_test']



# %% Network

input_qBOLD = keras.Input(shape=(30,30,16,1), name = 'Input_qBOLD')

input_qBOLD.shape
input_qBOLD.dtype

n=8

conv_qBOLD_p1 = keras.layers.Conv3D(n,
                  kernel_size = [1,1,9],
                  strides=1,
                  padding='valid',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_qBOLD_p1')(input_qBOLD)

conv_qBOLD_p2 = keras.layers.Conv3D(2*n,
                  kernel_size = [1,1,8],
                  strides=1,
                  padding='valid',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_qBOLD_p2')(conv_qBOLD_p1)


conv_qBOLD_d1 = keras.layers.Conv3D(n,
                  (3,3,1),
                  strides=(2,2,1),
                  padding='same',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_qBOLD_d1')(input_qBOLD)

upSamp_qBOLD_1 = keras.layers.UpSampling3D(size=(2,2,1), name = 'upSamp_qBOLD_1'                   )(conv_qBOLD_d1)


conv_qBOLD_d2 = keras.layers.Conv3D(2*n,
                  kernel_size = [1,1,16],
                  strides=1,
                  padding='valid',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_qBOLD_d2')(upSamp_qBOLD_1)

concatenate_qBOLD = layers.Concatenate(name = 'Concat_qBOLD')([conv_qBOLD_p2,conv_qBOLD_d2])
dense_layer_qBOLD = layers.Dense(16,name = 'Dense_qBOLD')(concatenate_qBOLD)
model_qBOLD = keras.Model(inputs=input_qBOLD, outputs = dense_layer_qBOLD, name="qBOLD model")
model_qBOLD.summary()
keras.utils.plot_model(model_qBOLD, show_shapes=True)
# %%

input_QSM = keras.Input(shape=(30,30,1,1), name = 'Input_QSM')
conv_QSM_1 = keras.layers.Conv3D(8,
                  (3,3,1),
                  strides=(1,1,1),
                  padding='same',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_QSM_1')(input_QSM)

conv_QSM_2 = keras.layers.Conv3D(8,
                  (3,3,1),
                  strides=(2,2,1),
                  padding='same',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_QSM_2')(conv_QSM_1)

upSamp_QSM_1 = keras.layers.UpSampling3D(size=(2,2,1), name = 'upSamp_QSM_1'                   )(conv_QSM_2)

concatenate_QSM = layers.Concatenate(name = 'Concat_QSM')([input_QSM,upSamp_QSM_1])
dense_layer_QSM = layers.Dense(8,name = 'Dense_QSM')(concatenate_QSM)
model_QSM = keras.Model(inputs=input_QSM, outputs = dense_layer_QSM, name="QSM model")
model_QSM.summary()
keras.utils.plot_model(model_QSM, show_shapes=True)
# %%
concatenate_layer = layers.Concatenate(name = 'Concat_QQ')([model_qBOLD.output,model_QSM.output])

dense_layer_1 = layers.Dense(48,name = 'Dense_1')(concatenate_layer)
#dense_layer_2 = layers.Dense(96,name = 'Dense_2')(dense_layer_1)


dense_layer_3a = layers.Dense(1,activation="sigmoid", name = 'Dense_3a_S0')(    dense_layer_1)
dense_layer_3b = layers.Dense(1,activation="sigmoid", name = 'Dense_3b_R2')(    dense_layer_1)
dense_layer_3c = layers.Dense(1,activation="sigmoid", name = 'Dense_3c_Y')(     dense_layer_1)
dense_layer_3d = layers.Dense(1,activation="sigmoid", name = 'Dense_3d_nu')(    dense_layer_1)
dense_layer_3e = layers.Dense(1,activation="sigmoid", name = 'Dense_3e_chi_nb')(dense_layer_1)

#before_lambda_model = keras.Model(input_layer, dense_layer_3, name="before_lambda_model")

Params_Layer = layers.concatenate(name = 'Output_Params',inputs=[dense_layer_3a,dense_layer_3b,dense_layer_3c,dense_layer_3d,dense_layer_3e],axis=-2)
#Params_Layer = layers.Concatenate(name = 'Output_Params')([dense_layer_3b,dense_layer_3c])
model_params = keras.Model(inputs=[input_qBOLD,input_QSM],outputs=Params_Layer,name="Params_model")
model_params.summary()
keras.utils.plot_model(model_params, show_shapes=True)


# %% Train Params model
opt = keras.optimizers.Adam(0.001, clipnorm=1.)
model_params.compile(
    #loss=keras.losses.MeanAbsolutePercentageError(),
    #loss=keras.losses.MeanSquaredLogarithmicError(),
    loss=keras.losses.MeanSquaredError(),
    #loss=tf.keras.losses.Huber(),
    optimizer=opt,
    metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
    #metrics=[tf.keras.metrics.MeanSquaredError()],
    #metrics=["accuracy"],
)

#model_params.compile(optimizer='sgd', loss=tf.keras.losses.CosineSimilarity(axis=-2))

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]


history = model_params.fit([qBOLD_training,QSM_training], Params_training , batch_size=200, epochs=1000, validation_split=0.2, callbacks=my_callbacks)
test_scores = model_params.evaluate([qBOLD_test,QSM_test], Params_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

#%%
#model_params.save("Model_2D_Params_before.h5")
#"""
# %%
model_params = keras.models.load_model("Model_2D_Params_before_qqbold.h5")
model_params.summary()

# %%
p = model_params.predict([qBOLD_test,QSM_test])


#%%
Number = 2
fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
ax = axes.ravel()
P0 = ax[0].imshow(Params_test[Number,:,:,0], cmap='Greys')
ax[0].title.set_text('a')
plt.colorbar(P0,ax=ax[0])
P1 = ax[1].imshow(Params_test[Number,:,:,1], cmap='Greys')
ax[1].title.set_text('b')
plt.colorbar(P1,ax=ax[1])
P2 = ax[2].imshow(Params_test[Number,:,:,2], cmap='Greys')
ax[2].title.set_text('c')
plt.colorbar(P2,ax=ax[2])
P3 = ax[3].imshow(Params_test[Number,:,:,3], cmap='Greys')
ax[3].title.set_text('d')
plt.colorbar(P3,ax=ax[3])
P4 = ax[4].imshow(Params_test[Number,:,:,4], cmap='Greys')
ax[4].title.set_text('e')
plt.colorbar(P4,ax=ax[4])
P5 = ax[5].imshow(p[Number,:,:,0], cmap='Greys')
plt.colorbar(P5,ax=ax[5])
P6 = ax[6].imshow(p[Number,:,:,1], cmap='Greys')
plt.colorbar(P6,ax=ax[6])
P7 = ax[7].imshow(p[Number,:,:,2], cmap='Greys')
plt.colorbar(P7,ax=ax[7])
P8 = ax[8].imshow(p[Number,:,:,3], cmap='Greys')
plt.colorbar(P8,ax=ax[8])
P9 = ax[9].imshow(p[Number,:,:,4], cmap='Greys')
plt.colorbar(P9,ax=ax[9])
plt.show()

# %%
fig, axes = plt.subplots(nrows=1, ncols=5,figsize=(15,5))
ax = axes.ravel()
ax[0].plot(Params_test[Number,15,:,0])
ax[0].plot(p[Number,15,:,0])
ax[0].title.set_text('a')
ax[1].plot(Params_test[Number,15,:,1])
ax[1].plot(p[Number,15,:,1])
ax[1].title.set_text('b')
ax[2].plot(Params_test[Number,15,:,2])
ax[2].plot(p[Number,15,:,2])
ax[2].title.set_text('c')
ax[3].plot(Params_test[Number,15,:,3])
ax[3].plot(p[Number,15,:,3])
ax[3].title.set_text('d')
ax[4].plot(Params_test[Number,15,:,4])
ax[4].plot(p[Number,15,:,4])
ax[4].title.set_text('e')
plt.show()

# %%

fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
ax = axes.ravel()
ax[0].hist(Params_test[Number,:,:,0].ravel())
ax[0].title.set_text('a')
ax[1].hist(Params_test[Number,:,:,1].ravel())
ax[1].title.set_text('b')
ax[2].hist(Params_test[Number,:,:,2].ravel())
ax[2].title.set_text('c')
ax[3].hist(Params_test[Number,:,:,3].ravel())
ax[3].title.set_text('d')
ax[4].hist(Params_test[Number,:,:,4].ravel())
ax[4].title.set_text('e')
ax[5].hist(p[Number,:,:,0].ravel())
ax[6].hist(p[Number,:,:,1].ravel())
ax[7].hist(p[Number,:,:,2].ravel())
ax[8].hist(p[Number,:,:,3].ravel())
ax[9].hist(p[Number,:,:,4].ravel())
plt.show()



# %%
""" Second training step """
# TODO: check values for t, still the same every 3 milli seconds
# TODO: translate Functions of QSM QBOLD to tf.math for FID, Echo Rise and Echo Fall


def f_hyper_tensor(x):
    '''
    Write hypergeometric function as taylor order 10 for beginning and as x-1 for larger numbers
    Exakt equation: hypergeom(-0.5,[0.75,1.25],-9/16*x.^2)-1
    (Intersection>x)*taylor + (x>=Intersection)*(x-1)
    taylor = - (81*x^8)/10890880 + (27*x^6)/80080 - (3*x^4)/280 + (3*x^2)/10
    Intersection at approx x = 3.72395
    '''
    Intersection = tf.constant(3.72395,dtype=tf.float32)
    a = -81./10890880*tf.math.pow(x,8) +27./80080*tf.math.pow(x,6) -3./280*tf.math.pow(x,4) +3./10*tf.math.pow(x,2)
    b = x-1
    return tf.where(tf.math.greater(Intersection,x),a, b)

def test_f_hyper_tensor():
    t=tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,40,50,60,70], dtype=tf.float32)/10
    out = f_hyper_tensor(t)
    plt.plot(t.numpy(),out.numpy(),'o-')

#test_f_hyper_tensor()

def simulateSignal_for_FID(tensor):
    t=tf.constant([3,6,9,12,15,18], dtype=tf.float32)/1000
    a = tensor[0]  #ln(S0)
    b = tensor[1]
    c = tensor[2]
    d = tensor[3]
    e = tensor[4]
    S0 = a   #S0     = 1000 + 200 * randn(N).T
    R2 = (30-1) * b + 1
    SaO2 = 0.98
    Y  = (SaO2 - 0.01) * c + 0.01
    nu = (0.1 - 0.001) * d + 0.001
    chi_nb = ( 0.1-(-0.1) ) * e - 0.1
    TE = 40/1000
    Hct = 0.357
    # Blood Hb volume fraction
    psi_Hb = Hct*0.34/1.335
    # Susceptibility of oxyhemoglobin in ppm
    chi_oHb = -0.813
    # Susceptibility of plasma in ppm
    chi_p = -0.0377
    # Susceptibility of fully oxygenated blood in ppm
    chi_ba = psi_Hb*chi_oHb + (1-psi_Hb)*chi_p
    #CF = gamma *B0
    gamma = 267.513 #MHz/T
    B0 = 3 #T
    delta_chi0 = 4*np.pi*0.273 #in ppm
    dw = 1./3 * gamma * B0* (Hct * delta_chi0 * (1-Y) + chi_ba - chi_nb )

    output = S0 * tf.math.exp(-R2*t - nu*f_hyper_tensor(dw*t))
    return output
Params_test.shape
p.shape
test_a = tf.convert_to_tensor(Params_test[2,:,:,0])
test_b = tf.convert_to_tensor(Params_test[2,:,:,1])
test_c = tf.convert_to_tensor(Params_test[2,:,:,2])
test_d = tf.convert_to_tensor(Params_test[2,:,:,3])
test_e = tf.convert_to_tensor(Params_test[2,:,:,4])

out = simulateSignal_for_FID([test_a,test_b,test_c,test_d,test_e])


def simulateSignal_for_Echo_Peak_rise(tensor):
    # x[0] = S0, x[1] = T2, x[2] = T2S
    t=tf.constant([21,24,27,30,33,36,39], dtype=tf.float32)/1000
    S0 = tensor[0]
    T2 = tensor[1]
    T2S = tensor[2]
    #output = S0 - (40.0-t)*(tf.math.divide_no_nan(1.0,T2S) - tf.math.divide_no_nan(1.0,T2)) - tf.math.divide_no_nan(t,T2)
    output = S0 * tf.math.exp(- (40.0-t)*(tf.math.divide_no_nan(1.0,T2S) - tf.math.divide_no_nan(1.0,T2)) - tf.math.divide_no_nan(t,T2) )
    return output

def simulateSignal_for_Echo_Peak_fall(tensor):
    # x[0] = S0, x[1] = T2, x[2] = T2S
    t=tf.constant([42,45,48], dtype=tf.float32)/1000
    S0 = tensor[0]
    T2 = tensor[1]
    T2S = tensor[2]
    #output = S0 - (t-40.0)*(tf.math.divide_no_nan(1.0,T2S) - tf.math.divide_no_nan(1.0,T2)) - tf.math.divide_no_nan(t,T2)
    output = S0 * tf.math.exp(- (t-40.0)*(tf.math.divide_no_nan(1.0,T2S) - tf.math.divide_no_nan(1.0,T2)) - tf.math.divide_no_nan(t,T2) )
    return output



#%%
FID_Layer = layers.Lambda(simulateSignal_for_FID, name = 'FID')([dense_layer_3a,dense_layer_3c])
Echo_Peak_rise_layer = layers.Lambda(simulateSignal_for_Echo_Peak_rise, name = 'SE_rise')([dense_layer_3a,dense_layer_3b,dense_layer_3c])
Echo_Peak_fall_layer = layers.Lambda(simulateSignal_for_Echo_Peak_fall, name = 'SE_fall')([dense_layer_3a,dense_layer_3b,dense_layer_3c])
output_layer = layers.Concatenate(name = 'Output_layer')([FID_Layer,Echo_Peak_rise_layer,Echo_Peak_fall_layer])


model = keras.Model(inputs=input_layer,outputs=output_layer,name="Lambda_model")
model.summary()
keras.utils.plot_model(model, show_shapes=True)

# %% Train full model
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer='adam',
    metrics=["accuracy"],
)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]

history = model.fit(input_noise_norm_train, input_noise_norm_train, batch_size=20, epochs=50, validation_split=0.2, callbacks=my_callbacks)
#test_scores = model.evaluate(input_noise_norm_test, signal_test, verbose=2)
test_scores = model.evaluate(input_noise_norm_test, input_noise_norm_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
test_scores_params = model_params.evaluate(input_noise_norm_test, target_test, verbose=2)

model_params.save("Model_2D_Params_after.h5")
model.save("Model_2D_Full.h5")


# %%
p_after = model_params.predict(input_noise_norm_test)

Number = 1

fig, axes = plt.subplots(nrows=2, ncols=3)
ax = axes.ravel()
ax[0].imshow(target_test[Number,:,:,0], cmap='Greys')
ax[0].title.set_text('S0')
ax[1].imshow(target_test[Number,:,:,1], cmap='Greys')
ax[1].title.set_text('T2')
ax[2].imshow(target_test[Number,:,:,2], cmap='Greys')
ax[2].title.set_text('T2S')
ax[3].imshow(p_after[Number,:,:,0], cmap='Greys')
#ax[3].title.set_text('S0')
ax[4].imshow(p_after[Number,:,:,1], cmap='Greys')
#ax[4].title.set_text('T2')
ax[5].imshow(p_after[Number,:,:,2], cmap='Greys')
#ax[5].title.set_text('T2S')
plt.show()
# %%
fig, axes = plt.subplots(nrows=1, ncols=3)
ax = axes.ravel()
ax[0].plot(target_test[Number,64,:,0])
ax[0].plot(p_after[Number,64,:,0])
ax[0].title.set_text('S0')
ax[1].plot(target_test[Number,64,:,1])
ax[1].plot(p_after[Number,64,:,1])
ax[1].title.set_text('T2')
ax[2].plot(target_test[Number,64,:,2])
ax[2].plot(p_after[Number,64,:,2])
ax[2].title.set_text('T2S')
plt.show()
#"""
