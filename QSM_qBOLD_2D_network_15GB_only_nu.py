# %% import modules
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
from numpy.random import rand, randn,shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm  #for progress bar
import levenberg_marquardt as lm
import h5py
#from QSM_qBOLD_2D_load_and_prepare_data import load_and_prepare_data

#tf.keras.mixed_precision.set_global_policy("mixed_float16") #accelerates training, expecially with tensor cores on RTX cards
#from My_Custom_Generator import My_Params_Generator,My_Signal_Generator
#%%
#data_dir = "../Brain_Phantom/Patches/"
#Params_training,Params_test,qBOLD_training,qBOLD_test,QSM_training,QSM_test = load_and_prepare_data(data_dir)

#np.savez("../Brain_Phantom/Patches/NumpyArchiv",Params_training=Params_training,Params_test=Params_test,qBOLD_training=qBOLD_training,qBOLD_test=qBOLD_test,QSM_training=QSM_training,QSM_test=QSM_test)

Dataset_train=np.load("../Brain_Phantom/Patches_no_air_big/15GB_1Pnoise_train_val.npz")
Dataset_test=np.load("../Brain_Phantom/Patches_no_air_big/15GB_1Pnoise_test.npz")
#S0_train=Dataset_train['S0']
#R2_train=Dataset_train['R2']
#Y_train=Dataset_train['Y']
nu_train=Dataset_train['nu']
#chi_nb_train=Dataset_train['chi_nb']

#S0_test=Dataset_test['S0']
#R2_test=Dataset_test['R2']
#Y_test=Dataset_test['Y']
nu_test=Dataset_test['nu']
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
Dataset_reduced=np.load("../Brain_Phantom/Patches_no_air_big/qBOLD_15GB_S0_R2_removed.npz")
qBOLD_training=Dataset_reduced['qBOLD_training']
qBOLD_test=Dataset_reduced['qBOLD_test']

QSM_training=Dataset_train['QSM']
QSM_test=Dataset_test['QSM']

version = "no_air_1Pnoise_15GB_only_nu/"
#%%
features_dataset=tf.data.Dataset.from_tensor_slices((qBOLD_training,QSM_training))
labels_dataset=tf.data.Dataset.from_tensor_slices(nu_train)
train_dataset=tf.data.Dataset.zip((features_dataset,labels_dataset))
batch_size=2000
train_dataset = train_dataset.batch(batch_size).cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# %% Network

input_qBOLD = keras.Input(shape=(30,30,16), name = 'Input_qBOLD')

input_qBOLD.shape
input_qBOLD.dtype

n=8

conv_qBOLD_1 = keras.layers.Conv2D(n,
                  kernel_size = 3,
                  strides=1,
                  padding='same',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_qBOLD_1')(input_qBOLD)

conv_qBOLD_2 = keras.layers.Conv2D(2*n,
                  kernel_size = 3,
                  strides=1,
                  padding='same',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_qBOLD_2')(conv_qBOLD_1)




model_qBOLD = keras.Model(inputs=input_qBOLD, outputs = conv_qBOLD_2, name="qBOLD model")
model_qBOLD.summary()
keras.utils.plot_model(model_qBOLD, show_shapes=True)
# %%

input_QSM = keras.Input(shape=(30,30,1), name = 'Input_QSM')
conv_QSM_1 = keras.layers.Conv2D(n,
                  kernel_size=3,
                  strides=(1),
                  padding='same',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_QSM_1')(input_QSM)
conv_QSM_2 = keras.layers.Conv2D(2*n,
                  kernel_size=3,
                  strides=(1),
                  padding='same',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_QSM_2')(conv_QSM_1)


model_QSM = keras.Model(inputs=input_QSM, outputs = conv_QSM_2, name="QSM model")
model_QSM.summary()
keras.utils.plot_model(model_QSM, show_shapes=True)
# %%
concat_QQ_1 = layers.Concatenate(name = 'concat_QQ_1')([input_qBOLD,input_QSM]) #model_qBOLD.output,model_QSM.output,

conv_QQ_1 = layers.Conv2D(4*n,3,padding='same',name = 'conv_QQ_1')(concat_QQ_1)
conv_QQ_2 = layers.Conv2D(8*n,3,padding='same',name = 'conv_QQ_2')(conv_QQ_1)

#concat_QQ_2 = layers.Concatenate(name = 'concat_QQ_2')([concat_QQ_1,conv_QQ_2])



#conv_S0 = layers.Conv2D(1,3,padding='same',activation="sigmoid", name = 'S0')(    conv_QQ_1)
#conv_R2 = layers.Conv2D(1,3,padding='same',activation="sigmoid", name = 'R2')(    conv_QQ_1)
#conv_Y = layers.Conv2D(1,3,padding='same',activation="sigmoid", name = 'Y')(     conv_QQ_1)
conv_nu = layers.Conv2D(1,3,padding='same',activation="sigmoid", name = 'nu')(    conv_QQ_2)
#conv_chinb = layers.Conv2D(1,3,padding='same',activation="sigmoid", name = 'chi_nb')(conv_QQ_1)


model_params = keras.Model(inputs=[input_qBOLD,input_QSM],outputs=[conv_nu],name="Params_model")
model_params.summary()
keras.utils.plot_model(model_params, show_shapes=True)


# %% Train Params model

opt = keras.optimizers.Adam(0.001, clipnorm=1.)
#loss=keras.losses.MeanAbsolutePercentageError()
#loss=keras.losses.MeanSquaredLogarithmicError()
loss=keras.losses.MeanSquaredError()
#loss=tf.keras.losses.Huber()
losses = {
    #"S0":loss,
    #"R2":loss,
    #"Y":loss,
    "nu":loss,
    #"chi_nb":loss,
}
lossWeights = {
    #"S0":1.0,
    #"R2":1.0,
    #"Y":1.0,
    "nu":1.0,
    #"chi_nb":1.0,
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


#training_list = [S0_train,R2_train,Y_train,nu_train,chi_nb_train]

#%% train normal

history_params = model_params.fit([qBOLD_training,QSM_training], nu_train , batch_size=2000, epochs=100, validation_split=0.1/0.9, callbacks=my_callbacks)
#history_params = model_params.fit(training_Params_data, epochs=100,validation_data=val_Params_data, callbacks=my_callbacks)
#%%
model_wrapper = lm.ModelWrapper(model_params)
model_wrapper.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
    loss=lm.MeanSquaredError())

history_params = model_wrapper.fit([qBOLD_training,QSM_training], nu_train , batch_size=2000, epochs=20, validation_split=0.1/0.9)


#%%
model_params.save("models/"+version+ "Model_2D_fully_conv_Params_S0_R2_removed.h5")
np.save('models/'+version+'history_params_2D_fully_conv_S0_R2_removed.npy',history_params.history)

#%%
#test_list = [S0_test,R2_test,Y_test,nu_test,chi_nb_test]

test_scores = model_params.evaluate([qBOLD_test,QSM_test], nu_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
#%%
def plot_loss(history, keyword):
    plt.figure()
    plt.plot(history.history[keyword + 'loss'],'o-')
    plt.plot(history.history['val_'+keyword+'loss'],'o-')
    plt.yscale('log')
    plt.title('model ' +keyword+ 'loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

#%%
print(history_params.history.keys())

plot_loss(history_params,'')


# %%
#model_params = keras.models.load_model("models/"+version+ "Model_2D_Params_before_qqbold.h5")
#model_params.summary()
p = model_params.predict([qBOLD_test,QSM_test])
p.shape

#%%
def check_Params(Params_test,p,Number):
    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(5,10))
    ax = axes.ravel()
    P0 = ax[0].imshow(Params_test[Number,:,:], cmap='gray')
    P0.set_clim(.0,1)
    ax[0].title.set_text('d')
    plt.colorbar(P0,ax=ax[0])
    P5 = ax[1].imshow(np.squeeze(p[Number,:,:,:]), cmap='gray')
    P5.set_clim(.0,1)
    plt.colorbar(P5,ax=ax[1])
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(5,5))
    ax.plot(Params_test[Number,15,:],'.')
    ax.plot(np.squeeze(p[Number,15,:,:]),'.')
    ax.set_ylim(0,1)
    ax.title.set_text('d')
    plt.show()


    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(5,10))
    ax = axes.ravel()
    ax[0].hist(Params_test[Number,:,:].ravel(),range=((0,1)))
    ax[0].title.set_text('d')
    ax[1].hist(np.squeeze(p[Number,:,:,:]).ravel(),range=((0,1)))
    plt.show()
#%%
Number = 5
check_Params(nu_test,p,Number)

#%%
def translate_Params(Params):
    nu = (0.1 - 0.001) * Params + 0.001  #from 0.1% to 10%
    return nu


def check_Params_transformed(Params_test,p,Number):
    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(5,10))
    ax = axes.ravel()
    P0 = ax[0].imshow(Params_test[Number,:,:], cmap='gray')
    P0.set_clim(.0,.1)
    ax[0].title.set_text('nu')
    plt.colorbar(P0,ax=ax[0])
    P5 = ax[1].imshow(np.squeeze(p[Number,:,:,:]), cmap='gray')
    P5.set_clim(.0,.1)
    plt.colorbar(P5,ax=ax[1])
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(5,5))
    ax.plot(Params_test[Number,15,:],'.')
    ax.plot(np.squeeze(p[Number,15,:,:]),'.')
    ax.set_ylim(0,.1)
    ax.title.set_text('nu')
    plt.show()


    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(5,10))
    ax = axes.ravel()
    ax[0].hist(Params_test[Number,:,:].ravel(),range=((0,.1)))
    ax[0].title.set_text('nu')
    ax[1].hist(np.squeeze(p[Number,:,:,:]).ravel(),range=((0,.1)))
    plt.show()

#%%
label_transformed=translate_Params(nu_test)
prediction_transforemd=translate_Params(p)
check_Params_transformed(label_transformed,prediction_transforemd,5)


#%%
Number=2
QSM_test.shape
plt.figure()
plt.imshow(QSM_test[Number,:,:,0], cmap='gray')

# %%
""" Second training step """

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
    output_FID = S0 * tf.math.exp(-R2*t_FID)*(1 - nu/(1-nu)*f_hyper_tensor(dw*t_FID) +1/(1-nu)*f_hyper_tensor(nu*dw*t_FID) )
    t_Echo_rise=tf.constant([21,24,27,30,33,36,39], dtype=tf.float32)/1000
    output_Echo_rise = S0 * tf.math.exp(-R2*t_Echo_rise)*(1 - nu/(1-nu)*f_hyper_tensor(dw*(TE-t_Echo_rise)) +1/(1-nu)*f_hyper_tensor(nu*dw*(TE-t_Echo_rise)) )
    t_Echo_fall=tf.constant([42,45,48], dtype=tf.float32)/1000
    output_Echo_fall = S0 * tf.math.exp(-R2*t_Echo_fall)*(1 - nu/(1-nu)*f_hyper_tensor(dw*(t_Echo_fall-TE)) +1/(1-nu)*f_hyper_tensor(nu*dw*(t_Echo_fall-TE)) )
    return tf.concat([output_FID,output_Echo_rise,output_Echo_fall],axis=-1)


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
#qBOLD_layer = layers.Lambda(f_qBOLD_tensor, name = 'qBOLD')([dense_layer_3a,dense_layer_3b,dense_layer_3c,dense_layer_3d,dense_layer_3e])
qBOLD_layer = layers.Lambda(f_qBOLD_tensor, name = 'qBOLD')(model_params.output)
#QSM_layer = layers.Lambda(f_QSM_tensor, name = 'QSM')([dense_layer_3c,dense_layer_3d,dense_layer_3e])
QSM_layer = layers.Lambda(f_QSM_tensor, name = 'QSM')(model_params.output[2:5])
model_params.output[2:5]

model = keras.Model(inputs=[input_qBOLD,input_QSM],outputs=[qBOLD_layer,QSM_layer],name="Lambda_model")
model.summary()
keras.utils.plot_model(model, show_shapes=True)

# %% Train full model
loss=keras.losses.MeanSquaredError()
losses = {
    "qBOLD":keras.losses.MeanAbsolutePercentageError(),
    "QSM":loss,
}
lossWeights = {
    "qBOLD":1.0/100,
    "QSM":1.0*10,
    }

model.compile(
    loss=losses,
    loss_weights=lossWeights,
    optimizer=opt,
    metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsolutePercentageError()],
    #metrics=["accuracy"],
)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3),
    tf.keras.callbacks.ModelCheckpoint(filepath='models/'+version+'model.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]
history = model.fit([qBOLD_training,QSM_training], [qBOLD_training,QSM_training] , batch_size=100, epochs=100, validation_split=0.1/0.9, callbacks=my_callbacks)

test_scores = model.evaluate([qBOLD_test,QSM_test],  [qBOLD_test,QSM_test], verbose=2)
#qBOLD_test.shape
#qBOLD_test2 = tf.expand_dims(qBOLD_test,axis=3)
#qBOLD_test2.shape
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
test_scores_params = model_params.evaluate([qBOLD_test,QSM_test], test_list, verbose=2)

#%%
model_params.save("models/"+version+ "Model_2D_fully_conv_Params_full_trained_different_fBOLD_MAPE.h5")
model.save("models/"+version+ "Model_2D_fully_conv_Full_different_fBOLD_MAPE.h5")

# %%
print(history.history.keys())
plot_loss(history,'')
plot_loss(history,'qBOLD_')
plot_loss(history,'QSM_')

#%%
p_after = model_params.predict([qBOLD_test,QSM_test])
#%%
Number = 2

check_Params(test_list,p_after,Number)

#%%
label_transformed=translate_Params(test_list)
prediction_transforemd=translate_Params(p_after)
check_Params_transformed(label_transformed,prediction_transforemd,Number)
#%%
def check_QSM(t,p,Number): #target prediction
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax = axes.ravel()
    P0=ax[0].imshow(t[Number,:,:,0], cmap='gray')
    ax[0].title.set_text('QSM')
    plt.colorbar(P0,ax=ax[0])
    P1=ax[1].imshow(p[Number,:,:,0], cmap='gray')
    #ax[1].title.set_text('QSM_pred')
    plt.colorbar(P1,ax=ax[1])
    plt.show()

p_full = model.predict([qBOLD_test,QSM_test])
QSM_test.shape
len(p_full)
p_full[1].shape
check_QSM(QSM_test,p_full[1],Number)


#%%
def check_qBOLD(t,p,Number): #target prediction
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()

    P0=ax[0].imshow(t[Number,:,:,0], cmap='gray')
    ax[0].title.set_text('3ms')
    #plt.colorbar(P0,ax=ax[0])
    P0.set_clim(.0,1)

    P1=ax[1].imshow(t[Number,:,:,3], cmap='gray')
    ax[1].title.set_text('9ms')
    #plt.colorbar(P1,ax=ax[1])
    P1.set_clim(.0,1)

    P2=ax[2].imshow(t[Number,:,:,7], cmap='gray')
    ax[2].title.set_text('21ms')
    #plt.colorbar(P2,ax=ax[2])
    P2.set_clim(.0,1)

    P3=ax[3].imshow(t[Number,:,:,11], cmap='gray')
    ax[3].title.set_text('33ms')
    #plt.colorbar(P3,ax=ax[3])
    P3.set_clim(.0,1)

    P4=ax[4].imshow(t[Number,:,:,15], cmap='gray')
    ax[4].title.set_text('45ms')
    plt.colorbar(P4,ax=ax[4])
    P4.set_clim(.0,1)

    P5=ax[5].imshow(p[Number,:,:,0], cmap='gray')
    #plt.colorbar(P5,ax=ax[5])
    P5.set_clim(.0,1)

    P6=ax[6].imshow(p[Number,:,:,3], cmap='gray')
    #plt.colorbar(P6,ax=ax[6])
    P6.set_clim(.0,1)

    P7=ax[7].imshow(p[Number,:,:,7], cmap='gray')
    #plt.colorbar(P7,ax=ax[7])
    P7.set_clim(.0,1)

    P8=ax[8].imshow(p[Number,:,:,11], cmap='gray')
    #plt.colorbar(P8,ax=ax[8])
    P8.set_clim(.0,1)

    P9=ax[9].imshow(p[Number,:,:,15], cmap='gray')
    plt.colorbar(P9,ax=ax[9])
    P9.set_clim(.0,1)
    plt.show()


qBOLD_test.shape
len(p_full)
p_full[0].shape
check_qBOLD(qBOLD_test,p_full[0],Number)


#%% check qBOLD Verlauf

def check_Pixel(target,prediction,QSM_t,QSM_p,Number):
    t=np.array([3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48])/1000
    plt.figure()
    plt.imshow(target[Number,:,:,0], cmap='gray')
    plt.plot([5,10,15,20,25],[15,15,15,15,15],'o')
    plt.show()
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,5,15,:],"o-r")
    ax[0].plot(t,prediction[Number,5,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,5,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,5,15,0],"ob")
    ax[1].set_ylim(-0.15,0.15)
    plt.show()
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,10,15,:],"o-r")
    ax[0].plot(t,prediction[Number,10,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,10,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,10,15,0],"ob")
    ax[1].set_ylim(-0.15,0.15)
    plt.show(    )
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,15,15,:],"o-r")
    ax[0].plot(t,prediction[Number,15,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,15,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,15,15,0],"ob")
    ax[1].set_ylim(-0.15,0.15)
    plt.show()
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,20,15,:],"o-r")
    ax[0].plot(t,prediction[Number,20,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,20,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,20,15,0],"ob")
    ax[1].set_ylim(-0.15,0.15)
    plt.show()
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,25,15,:],"o-r")
    ax[0].plot(t,prediction[Number,25,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,25,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,25,15,0],"ob")
    ax[1].set_ylim(-0.15,0.15)
    plt.show()

Number=2
check_Pixel(qBOLD_test,p_full[0],QSM_test,p_full[1],Number)
