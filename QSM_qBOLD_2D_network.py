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


dense_layer_3a = layers.Dense(1,activation="sigmoid", name = 'output_S0')(    dense_layer_1)
dense_layer_3b = layers.Dense(1,activation="sigmoid", name = 'output_R2')(    dense_layer_1)
dense_layer_3c = layers.Dense(1,activation="sigmoid", name = 'output_Y')(     dense_layer_1)
dense_layer_3d = layers.Dense(1,activation="sigmoid", name = 'output_nu')(    dense_layer_1)
dense_layer_3e = layers.Dense(1,activation="sigmoid", name = 'output_chi_nb')(dense_layer_1)

#before_lambda_model = keras.Model(input_layer, dense_layer_3, name="before_lambda_model")

Params_Layer = layers.concatenate(name = 'Output_Params',inputs=[dense_layer_3a,dense_layer_3b,dense_layer_3c,dense_layer_3d,dense_layer_3e],axis=-2)
#Params_Layer = layers.Concatenate(name = 'Output_Params')([dense_layer_3b,dense_layer_3c])
model_params = keras.Model(inputs=[input_qBOLD,input_QSM],outputs=[dense_layer_3a,dense_layer_3b,dense_layer_3c,dense_layer_3d,dense_layer_3e],name="Params_model")
model_params.summary()
keras.utils.plot_model(model_params, show_shapes=True)


# %% Train Params model

opt = keras.optimizers.Adam(0.001, clipnorm=1.)
loss=keras.losses.MeanAbsolutePercentageError()
#loss=keras.losses.MeanSquaredLogarithmicError()
#loss=keras.losses.MeanSquaredError()
#loss=tf.keras.losses.Huber()
losses = {
    "output_S0":loss,
    "output_R2":loss,
    "output_Y":loss,
    "output_nu":loss,
    "output_chi_nb":loss,
}
lossWeights = {
    "output_S0":1.0,
    "output_R2":1.0,
    "output_Y":1.0,
    "output_nu":1.0,
    "output_chi_nb":1.0,
}
model_params.compile(
    loss=losses,
    loss_weights=lossWeights,
    optimizer=opt,
    metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
    #metrics=[tf.keras.metrics.MeanSquaredError()],
    #metrics=["accuracy"],
)


my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]


S0_training = tf.expand_dims(Params_training[:,:,:,0],-1)
R2_training = tf.expand_dims(Params_training[:,:,:,1],-1)
Y_training = tf.expand_dims(Params_training[:,:,:,2],-1)
nu_training = tf.expand_dims(Params_training[:,:,:,3],-1)
chi_nb_training = tf.expand_dims(Params_training[:,:,:,4],-1)
target_list = [S0_training,R2_training,Y_training,nu_training,chi_nb_training]

history = model_params.fit([qBOLD_training,QSM_training], target_list , batch_size=200, epochs=1000, validation_split=0.2, callbacks=my_callbacks)
test_scores = model_params.evaluate([qBOLD_test,QSM_test], Params_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

#%%
#model_params.save("models/Model_2D_Params_before_qqbold.h5")

# %%
model_params = keras.models.load_model("Model_2D_Params_before_qqbold.h5")
#model_params.summary()
p = model_params.predict([qBOLD_test,QSM_test])
p[0].shape
#%%
def check_Params(Params_test,p):
    Number = 2
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    P0 = ax[0].imshow(Params_test[Number,:,:,0], cmap='gray')
    ax[0].title.set_text('a')
    plt.colorbar(P0,ax=ax[0])
    P1 = ax[1].imshow(Params_test[Number,:,:,1], cmap='gray')
    ax[1].title.set_text('b')
    plt.colorbar(P1,ax=ax[1])
    P2 = ax[2].imshow(Params_test[Number,:,:,2], cmap='gray')
    ax[2].title.set_text('c')
    plt.colorbar(P2,ax=ax[2])
    P3 = ax[3].imshow(Params_test[Number,:,:,3], cmap='gray')
    ax[3].title.set_text('d')
    plt.colorbar(P3,ax=ax[3])
    P4 = ax[4].imshow(Params_test[Number,:,:,4], cmap='gray')
    ax[4].title.set_text('e')
    plt.colorbar(P4,ax=ax[4])
    P5 = ax[5].imshow(np.squeeze(p[0][Number,:,:,:]), cmap='gray')
    plt.colorbar(P5,ax=ax[5])
    P6 = ax[6].imshow(np.squeeze(p[1][Number,:,:,:]), cmap='gray')
    plt.colorbar(P6,ax=ax[6])
    P7 = ax[7].imshow(np.squeeze(p[2][Number,:,:,:]), cmap='gray')
    plt.colorbar(P7,ax=ax[7])
    P8 = ax[8].imshow(np.squeeze(p[3][Number,:,:,:]), cmap='gray')
    plt.colorbar(P8,ax=ax[8])
    P9 = ax[9].imshow(np.squeeze(p[4][Number,:,:,:]), cmap='gray')
    plt.colorbar(P9,ax=ax[9])
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].plot(Params_test[Number,15,:,0])
    ax[0].plot(np.squeeze(p[0][Number,15,:,:]))
    ax[0].title.set_text('a')
    ax[1].plot(Params_test[Number,15,:,1])
    ax[1].plot(np.squeeze(p[1][Number,15,:,:]))
    ax[1].title.set_text('b')
    ax[2].plot(Params_test[Number,15,:,2])
    ax[2].plot(np.squeeze(p[2][Number,15,:,:]))
    ax[2].title.set_text('c')
    ax[3].plot(Params_test[Number,15,:,3])
    ax[3].plot(np.squeeze(p[3][Number,15,:,:]))
    ax[3].title.set_text('d')
    ax[4].plot(Params_test[Number,15,:,4])
    ax[4].plot(np.squeeze(p[4][Number,15,:,:]))
    ax[4].title.set_text('e')
    plt.show()


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
    ax[5].hist(np.squeeze(p[0][Number,:,:,:]).ravel())
    ax[6].hist(np.squeeze(p[1][Number,:,:,:]).ravel())
    ax[7].hist(np.squeeze(p[2][Number,:,:,:]).ravel())
    ax[8].hist(np.squeeze(p[3][Number,:,:,:]).ravel())
    ax[9].hist(np.squeeze(p[4][Number,:,:,:]).ravel())
    plt.show()

check_Params(Params_test,p)


#%%
Number=2
QSM_test.shape
plt.figure()
plt.imshow(QSM_test[Number,:,:,0], cmap='gray')

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

def f_qBOLD_tensor(tensor):
    a = tensor[0]
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

    TE=40./1000
    t_FID=tf.constant([3,6,9,12,15,18], dtype=tf.float32)/1000
    output_FID = S0 * tf.math.exp(-R2*t_FID - nu*f_hyper_tensor(dw*t_FID))
    t_Echo_rise=tf.constant([21,24,27,30,33,36,39], dtype=tf.float32)/1000
    output_Echo_rise = S0 * tf.math.exp(-R2*t_Echo_rise - nu*f_hyper_tensor(dw*(TE-t_Echo_rise)))
    t_Echo_fall=tf.constant([42,45,48], dtype=tf.float32)/1000
    output_Echo_fall = S0 * tf.math.exp(-R2*t_Echo_fall - nu*f_hyper_tensor(dw*(t_Echo_fall-TE)))
    return tf.concat([output_FID,output_Echo_rise,output_Echo_fall],axis=-1)
"""
This kind of test does not work because of Dimensions, but it works somehow in the network
Incompatible shapes: [30,30] vs. [6] [Op:Mul]


Params_test.shape
p.shape
test_a = tf.convert_to_tensor(Params_test[2,:,:,0])
test_a.shape
test_b = tf.convert_to_tensor(Params_test[2,:,:,1])
test_c = tf.convert_to_tensor(Params_test[2,:,:,2])
test_d = tf.convert_to_tensor(Params_test[2,:,:,3])
test_e = tf.convert_to_tensor(Params_test[2,:,:,4])

out = simulateSignal_for_FID([test_a,test_b,test_c,test_d,test_e])
"""

def f_QSM_tensor(tensor):
    c = tensor[0]
    d = tensor[1]
    e = tensor[2]

    Hct = 0.357
    SaO2 = 0.98
    # Ratio of deoxygenated and total blood volume
    alpha = 0.77;
    # Susceptibility difference between dHb and Hb in ppm
    delta_chi_Hb = 12.522;
    # Blood Hb volume fraction
    psi_Hb = Hct*0.34/1.335
    # Susceptibility of oxyhemoglobin in ppm
    chi_oHb = -0.813
    # Susceptibility of plasma in ppm
    chi_p = -0.0377
    # Susceptibility of fully oxygenated blood in ppm
    chi_ba = psi_Hb*chi_oHb + (1-psi_Hb)*chi_p

    Y  = (SaO2 - 0.01) * c + 0.01
    nu = (0.1 - 0.001) * d + 0.001
    chi_nb = ( 0.1-(-0.1) ) * e - 0.1

    Summand1 = (chi_ba/alpha +psi_Hb*delta_chi_Hb * ((1-(1-alpha)*SaO2)/alpha - Y) )*nu
    Summand2 = (1 - nu/alpha) * chi_nb

    return Summand1+Summand2 #np.array version is np.array([a+b]).T, maybe transpose here too


#%%
qBOLD_layer = layers.Lambda(f_qBOLD_tensor, name = 'qBOLD')([dense_layer_3a,dense_layer_3b,dense_layer_3c,dense_layer_3d,dense_layer_3e])
QSM_layer = layers.Lambda(f_QSM_tensor, name = 'QSM')([dense_layer_3c,dense_layer_3d,dense_layer_3e])


model = keras.Model(inputs=[input_qBOLD,input_QSM],outputs=[qBOLD_layer,QSM_layer],name="Lambda_model")
model.summary()
keras.utils.plot_model(model, show_shapes=True)

# %% Train full model
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer='adam',
    metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
    #metrics=["accuracy"],
)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]
history = model.fit([qBOLD_training,QSM_training], [tf.expand_dims(qBOLD_training,axis=3),tf.expand_dims(QSM_training,axis=3)] , batch_size=200, epochs=1000, validation_split=0.2, callbacks=my_callbacks)

test_scores = model.evaluate([qBOLD_test,QSM_test],  [tf.expand_dims(qBOLD_test,axis=3),tf.expand_dims(QSM_test,axis=3)], verbose=2)
#qBOLD_test.shape
#qBOLD_test2 = tf.expand_dims(qBOLD_test,axis=3)
#qBOLD_test2.shape
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
test_scores_params = model_params.evaluate([qBOLD_test,QSM_test], Params_test, verbose=2)

model_params.save("models/Model_2D_Params_after_qqbold.h5")
model.save("models/Model_2D_Full_qqbold.h5")


# %%

p_after = model_params.predict([qBOLD_test,QSM_test])

Number = 2

check_Params(Params_test,p_after)

#%%
def check_QSM(t,p): #target prediction
    Number = 2
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax = axes.ravel()
    P0=ax[0].imshow(t[Number,:,:,0], cmap='gray')
    ax[0].title.set_text('QSM')
    plt.colorbar(P0,ax=ax[0])
    P1=ax[1].imshow(p[Number,:,:,0,0], cmap='gray')
    #ax[1].title.set_text('QSM_pred')
    plt.colorbar(P1,ax=ax[1])
    plt.show()

p_full = model.predict([qBOLD_test,QSM_test])
QSM_test.shape
len(p_full)
p_full[1].shape
check_QSM(QSM_test,p_full[1])


#%%
def check_qBOLD(t,p): #target prediction
    Number = 2
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()

    P0=ax[0].imshow(t[Number,:,:,0], cmap='gray')
    ax[0].title.set_text('3ms')
    #plt.colorbar(P0,ax=ax[0])
    P0.set_clim(.0,.6)

    P1=ax[1].imshow(t[Number,:,:,3], cmap='gray')
    ax[1].title.set_text('9ms')
    #plt.colorbar(P1,ax=ax[1])
    P1.set_clim(.0,.6)

    P2=ax[2].imshow(t[Number,:,:,7], cmap='gray')
    ax[2].title.set_text('21ms')
    #plt.colorbar(P2,ax=ax[2])
    P2.set_clim(.0,.6)

    P3=ax[3].imshow(t[Number,:,:,11], cmap='gray')
    ax[3].title.set_text('33ms')
    #plt.colorbar(P3,ax=ax[3])
    P3.set_clim(.0,.6)

    P4=ax[4].imshow(t[Number,:,:,15], cmap='gray')
    ax[4].title.set_text('45ms')
    plt.colorbar(P4,ax=ax[4])
    P4.set_clim(.0,.6)

    P5=ax[5].imshow(p[Number,:,:,0,0], cmap='gray')
    #plt.colorbar(P5,ax=ax[5])
    P5.set_clim(.0,.6)

    P6=ax[6].imshow(p[Number,:,:,0,3], cmap='gray')
    #plt.colorbar(P6,ax=ax[6])
    P6.set_clim(.0,.6)

    P7=ax[7].imshow(p[Number,:,:,0,7], cmap='gray')
    #plt.colorbar(P7,ax=ax[7])
    P7.set_clim(.0,.6)

    P8=ax[8].imshow(p[Number,:,:,0,11], cmap='gray')
    #plt.colorbar(P8,ax=ax[8])
    P8.set_clim(.0,.6)

    P9=ax[9].imshow(p[Number,:,:,0,15], cmap='gray')
    plt.colorbar(P9,ax=ax[9])
    P9.set_clim(.0,.6)
    plt.show()


qBOLD_test.shape
len(p_full)
p_full[0].shape
check_qBOLD(qBOLD_test,p_full[0])
