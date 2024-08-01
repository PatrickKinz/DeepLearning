# %% import modules
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
from numpy.random import rand, randn,shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm  #for progress bar

import h5py

from qqboldparamsloadernoisy import QQBoldParamsLoader

#tf.keras.mixed_precision.set_global_policy("mixed_float16") #accelerates training, expecially with tensor cores on RTX cards
#from My_Custom_Generator import My_Params_Generator,My_Signal_Generator
#%%
#data_dir = "../Brain_Phantom/Patches/"
#Params_training,Params_test,qBOLD_training,qBOLD_test,QSM_training,QSM_test = load_and_prepare_data(data_dir)

#np.savez("../Brain_Phantom/Patches/NumpyArchiv",Params_training=Params_training,Params_test=Params_test,qBOLD_training=qBOLD_training,qBOLD_test=qBOLD_test,QSM_training=QSM_training,QSM_test=QSM_test)
"""
Dataset=np.load("../Brain_Phantom/Patches_no_air/NumpyArchiv_0noise.npz")
Params_training=Dataset['Params_training']
Params_test=Dataset['Params_test']
qBOLD_training=Dataset['qBOLD_training']
qBOLD_test=Dataset['qBOLD_test']
QSM_training=Dataset['QSM_training']
QSM_test=Dataset['QSM_test']
"""


#%%
version = "no_air_1Percentnoise_big/"

filenames=[]
filenumber=int(848820)
for count in range(filenumber):
    filenames.append("{0}.npz".format(count).zfill(6+4))
filenames_shuffled=shuffle(filenames)
#split
threshold1 = int(filenumber*0.8)
threshold2 = int(filenumber*0.9)
filenames_train_shuffled=filenames[:threshold1]
filenames_val_shuffled=filenames[threshold1:threshold2]
filenames_test_shuffled=filenames[threshold2:]
len(filenames_train_shuffled)
#filenames_val_shuffled
# %% Network

input_qBOLD = keras.Input(shape=(30,30,16), name = 'Input_qBOLD')

input_qBOLD.shape
input_qBOLD.dtype

n=16

conv_qBOLD_1 = keras.layers.Conv2D(n,
                  kernel_size = 3,
                  strides=1,
                  padding='same',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_qBOLD_1')(input_qBOLD)

"""
conv_qBOLD_2 = keras.layers.Conv2D(2*n,
                  kernel_size = 3,
                  strides=1,
                  padding='same',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_qBOLD_2')(conv_qBOLD_1)



concatenate_qBOLD = layers.Concatenate(name = 'Concat_qBOLD')([input_qBOLD,conv_qBOLD_2])
conv_qBOLD_3 =keras.layers.Conv2D(3*n,
                  kernel_size = 3,
                  strides=1,
                  padding='same',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_qBOLD_3')(concatenate_qBOLD)


"""
model_qBOLD = keras.Model(inputs=input_qBOLD, outputs = conv_qBOLD_1, name="qBOLD model")
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
"""
conv_QSM_2 = keras.layers.Conv2D(2*n,
                  kernel_size=3,
                  strides=(1),
                  padding='same',
                  dilation_rate=1,
                  activation='sigmoid',
                  name='conv_QSM_2')(conv_QSM_1)


concatenate_QSM = layers.Concatenate(name = 'Concat_QSM')([input_QSM,conv_QSM_2])
conv_QSM_3 = layers.Conv2D(3*n,3,padding='same',name = 'conv_QSM_3')(concatenate_QSM)
"""
model_QSM = keras.Model(inputs=input_QSM, outputs = conv_QSM_1, name="QSM model")
model_QSM.summary()
keras.utils.plot_model(model_QSM, show_shapes=True)
# %%
concat_QQ_1 = layers.Concatenate(name = 'concat_QQ_1')([model_qBOLD.output,model_QSM.output])

conv_QQ_1 = layers.Conv2D(2*n,3,padding='same',name = 'conv_QQ_1')(concat_QQ_1)
#conv_QQ_2 = layers.Conv2D(2*48,3,padding='same',name = 'conv_QQ_2')(conv_QQ_1)

#concat_QQ_2 = layers.Concatenate(name = 'concat_QQ_2')([concat_QQ_1,conv_QQ_2])



conv_S0 = layers.Conv2D(1,3,padding='same',activation="sigmoid", name = 'S0')(    conv_QQ_1)
conv_R2 = layers.Conv2D(1,3,padding='same',activation="sigmoid", name = 'R2')(    conv_QQ_1)
conv_Y = layers.Conv2D(1,3,padding='same',activation="sigmoid", name = 'Y')(     conv_QQ_1)
conv_nu = layers.Conv2D(1,3,padding='same',activation="sigmoid", name = 'nu')(    conv_QQ_1)
conv_chinb = layers.Conv2D(1,3,padding='same',activation="sigmoid", name = 'chi_nb')(conv_QQ_1)


model_params = keras.Model(inputs=[input_qBOLD,input_QSM],outputs=[conv_S0,conv_R2,conv_Y,conv_nu,conv_chinb],name="Params_model")
model_params.summary()
keras.utils.plot_model(model_params, show_shapes=True)


# %% Train Params model

opt = keras.optimizers.Adam(0.001, clipnorm=1.)
#loss=keras.losses.MeanAbsolutePercentageError()
#loss=keras.losses.MeanSquaredLogarithmicError()
loss=keras.losses.MeanSquaredError()
#loss=tf.keras.losses.Huber()
losses = {
    "S0":loss,
    "R2":loss,
    "Y":loss,
    "nu":loss,
    "chi_nb":loss,
}
lossWeights = {
    "S0":1.0,
    "R2":1.0,
    "Y":5.0,
    "nu":5.0,
    "chi_nb":1.0,
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
    tf.keras.callbacks.EarlyStopping(patience=3),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]

#%%

batch_size=100
train_params_loader = QQBoldParamsLoader(name="training_loader",shuffle=False,n_inputs=2,n_labels=5)
training_Params_data =train_params_loader(filenames_train_shuffled,batch_size=batch_size,read_threads=1)
val_params_loader = QQBoldParamsLoader(name="val_loader",shuffle=False,n_inputs=2,n_labels=5)
val_Params_data =train_params_loader(filenames_val_shuffled,batch_size=batch_size,read_threads=1)

#data_read = train_params_loader._read_file_and_return_numpy_samples(bytes(filenames_train_shuffled[0],'utf-8'))
#for i in data_read:
#    print(i.shape)

#id_tensor0 = tf.squeeze(tf.convert_to_tensor(filenames_train_shuffled[0], dtype=tf.string))
#id_tensor0
#id_tensor1 = tf.squeeze(tf.convert_to_tensor(filenames_train_shuffled[1], dtype=tf.string))
#id_tensor1
#id_tensor=[id_tensor0,id_tensor1]
#id_tensor
#file_list_ds = tf.data.Dataset.from_tensor_slices(id_tensor)
#test_read_wrapper = train_params_loader._read_wrapper(id_tensor0)
#test_read_wrapper
#test=training_Params_data.take(1)
#for element in test:
#    print(element)

#history_params = model_params.fit([qBOLD_training,QSM_training], training_list , batch_size=100, epochs=1000, validation_split=0.2, callbacks=my_callbacks)
#steps_per_epoch = len(filenames_train_shuffled)//batch_size,
#validation_steps = len(filenames_val_shuffled)//batch_size,
history_params = model_params.fit(training_Params_data, validation_data=val_Params_data, epochs = 100, callbacks=my_callbacks)

#%%
np.save('models/'+version+'history_params_2D_fully_conv_Params_before_qqbold_5_times_weights.npy',history_params.history)
np.save('models/'+version+'filenames_test_2D_fully_conv_Params_before_qqbold_5_times_weights.npy',filenames_test_shuffled)
test_params_loader = QQBoldParamsLoader(name="test_loader",shuffle=False,n_inputs=2,n_labels=5)
test_Params_data =train_params_loader(filenames_test_shuffled,batch_size=batch_size,read_threads=1)


test_scores = model_params.evaluate(test_Params_data, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

#%%
print(history_params.history.keys())
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

plot_loss(history_params,'')
plot_loss(history_params,'S0_')
plot_loss(history_params,'Y_')
plot_loss(history_params,'nu_')
plot_loss(history_params,'chi_nb_')

#%%
model_params.save("models/"+version+ "Model_2D_fully_conv_Params_before_qqbold_5_times_weights.h5")

# %%
#model_params = keras.models.load_model("models/"+version+ "Model_2D_Params_before_qqbold.h5")
#model_params.summary()
data_Params_reduced = test_Params_data.take(1)
data_element=list(data_Params_reduced.as_numpy_iterator())[0]
len(data_element) #2 x and y
len(data_element[1]) #5 the 5 parameters
data_element[1][0].shape
p = model_params.predict(data_element)
len(p)
p[0].shape

#%%
def check_Params(Params_test,p,Number):
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    P0 = ax[0].imshow(Params_test[0][Number,:,:], cmap='gray')
    P0.set_clim(.0,1)
    ax[0].title.set_text('a')
    plt.colorbar(P0,ax=ax[0])
    P1 = ax[1].imshow(Params_test[1][Number,:,:], cmap='gray')
    P1.set_clim(.0,1)
    ax[1].title.set_text('b')
    plt.colorbar(P1,ax=ax[1])
    P2 = ax[2].imshow(Params_test[2][Number,:,:], cmap='gray')
    P2.set_clim(.0,1)
    ax[2].title.set_text('c')
    plt.colorbar(P2,ax=ax[2])
    P3 = ax[3].imshow(Params_test[3][Number,:,:], cmap='gray')
    P3.set_clim(.0,1)
    ax[3].title.set_text('d')
    plt.colorbar(P3,ax=ax[3])
    P4 = ax[4].imshow(Params_test[4][Number,:,:], cmap='gray')
    P4.set_clim(.0,1)
    ax[4].title.set_text('e')
    plt.colorbar(P4,ax=ax[4])
    P5 = ax[5].imshow(np.squeeze(p[0][Number,:,:,:]), cmap='gray')
    P5.set_clim(.0,1)
    plt.colorbar(P5,ax=ax[5])
    P6 = ax[6].imshow(np.squeeze(p[1][Number,:,:,:]), cmap='gray')
    P6.set_clim(.0,1)
    plt.colorbar(P6,ax=ax[6])
    P7 = ax[7].imshow(np.squeeze(p[2][Number,:,:,:]), cmap='gray')
    P7.set_clim(.0,1)
    plt.colorbar(P7,ax=ax[7])
    P8 = ax[8].imshow(np.squeeze(p[3][Number,:,:,:]), cmap='gray')
    P8.set_clim(.0,1)
    plt.colorbar(P8,ax=ax[8])
    P9 = ax[9].imshow(np.squeeze(p[4][Number,:,:,:]), cmap='gray')
    P9.set_clim(.0,1)
    plt.colorbar(P9,ax=ax[9])
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].plot(Params_test[0][Number,15,:],'.')
    ax[0].plot(np.squeeze(p[0][Number,15,:,:]),'.')
    ax[0].set_ylim(0,1)
    ax[0].title.set_text('a')
    ax[1].plot(Params_test[1][Number,15,:],'.')
    ax[1].plot(np.squeeze(p[1][Number,15,:,:]),'.')
    ax[1].set_ylim(0,1)
    ax[1].title.set_text('b')
    ax[2].plot(Params_test[2][Number,15,:],'.')
    ax[2].plot(np.squeeze(p[2][Number,15,:,:]),'.')
    ax[2].set_ylim(0,1)
    ax[2].title.set_text('c')
    ax[3].plot(Params_test[3][Number,15,:],'.')
    ax[3].plot(np.squeeze(p[3][Number,15,:,:]),'.')
    ax[3].set_ylim(0,1)
    ax[3].title.set_text('d')
    ax[4].plot(Params_test[4][Number,15,:],'.')
    ax[4].plot(np.squeeze(p[4][Number,15,:,:]),'.')
    ax[4].set_ylim(0,1)
    ax[4].title.set_text('e')
    plt.show()


    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].hist(Params_test[0][Number,:,:].ravel(),range=((0,1)))
    ax[0].title.set_text('a')
    ax[1].hist(Params_test[1][Number,:,:].ravel(),range=((0,1)))
    ax[1].title.set_text('b')
    ax[2].hist(Params_test[2][Number,:,:].ravel(),range=((0,1)))
    ax[2].title.set_text('c')
    ax[3].hist(Params_test[3][Number,:,:].ravel(),range=((0,1)))
    ax[3].title.set_text('d')
    ax[4].hist(Params_test[4][Number,:,:].ravel(),range=((0,1)))
    ax[4].title.set_text('e')
    ax[5].hist(np.squeeze(p[0][Number,:,:,:]).ravel(),range=((0,1)))
    ax[6].hist(np.squeeze(p[1][Number,:,:,:]).ravel(),range=((0,1)))
    ax[7].hist(np.squeeze(p[2][Number,:,:,:]).ravel(),range=((0,1)))
    ax[8].hist(np.squeeze(p[3][Number,:,:,:]).ravel(),range=((0,1)))
    ax[9].hist(np.squeeze(p[4][Number,:,:,:]).ravel(),range=((0,1)))
    plt.show()
#%%
Number = 5
check_Params(data_element[1],p,Number)

#%%
def translate_Params(Params):
    S0 = Params[0]   #S0     = 1000 + 200 * randn(N).T
    R2 = (30-1) * Params[1] + 1  #from 1 to 30
    SaO2 = 0.98
    Y  = (SaO2 - 0.01) * Params[2] + 0.01   #from 1% to 98%
    nu = (0.1 - 0.001) * Params[3] + 0.001  #from 0.1% to 10%
    chi_nb = ( 0.1-(-0.1) ) * Params[4] - 0.1 #fr
    return [S0,R2,Y,nu,chi_nb]


def check_Params_transformed(Params_test,p,Number):
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    P0 = ax[0].imshow(Params_test[0][Number,:,:], cmap='gray')
    P0.set_clim(.0,1)
    ax[0].title.set_text('S0')
    plt.colorbar(P0,ax=ax[0])
    P1 = ax[1].imshow(Params_test[1][Number,:,:], cmap='gray')
    P1.set_clim(0,30)
    ax[1].title.set_text('R2')
    plt.colorbar(P1,ax=ax[1])
    P2 = ax[2].imshow(Params_test[2][Number,:,:], cmap='gray')
    P2.set_clim(.0,1)
    ax[2].title.set_text('Y')
    plt.colorbar(P2,ax=ax[2])
    P3 = ax[3].imshow(Params_test[3][Number,:,:], cmap='gray')
    P3.set_clim(0,0.1)
    ax[3].title.set_text('nu')
    plt.colorbar(P3,ax=ax[3])
    P4 = ax[4].imshow(Params_test[4][Number,:,:], cmap='gray')
    P4.set_clim(-.1,.1)
    ax[4].title.set_text('chi_nb')
    plt.colorbar(P4,ax=ax[4])
    P5 = ax[5].imshow(np.squeeze(p[0][Number,:,:,:]), cmap='gray')
    P5.set_clim(.0,1)
    plt.colorbar(P5,ax=ax[5])
    P6 = ax[6].imshow(np.squeeze(p[1][Number,:,:,:]), cmap='gray')
    P6.set_clim(0,30)
    plt.colorbar(P6,ax=ax[6])
    P7 = ax[7].imshow(np.squeeze(p[2][Number,:,:,:]), cmap='gray')
    P7.set_clim(.0,1)
    plt.colorbar(P7,ax=ax[7])
    P8 = ax[8].imshow(np.squeeze(p[3][Number,:,:,:]), cmap='gray')
    P8.set_clim(.0,0.1)
    plt.colorbar(P8,ax=ax[8])
    P9 = ax[9].imshow(np.squeeze(p[4][Number,:,:,:]), cmap='gray')
    P9.set_clim(-.1,.1)
    plt.colorbar(P9,ax=ax[9])
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].plot(Params_test[0][Number,15,:],'.')
    ax[0].plot(np.squeeze(p[0][Number,15,:,:]),'.')
    ax[0].set_ylim(0,1)
    ax[0].title.set_text('S0')
    ax[1].plot(Params_test[1][Number,15,:],'.')
    ax[1].plot(np.squeeze(p[1][Number,15,:,:]),'.')
    ax[1].set_ylim(0,30)
    ax[1].title.set_text('R2')
    ax[2].plot(Params_test[2][Number,15,:],'.')
    ax[2].plot(np.squeeze(p[2][Number,15,:,:]),'.')
    ax[2].set_ylim(0,1)
    ax[2].title.set_text('Y')
    ax[3].plot(Params_test[3][Number,15,:],'.')
    ax[3].plot(np.squeeze(p[3][Number,15,:,:]),'.')
    ax[3].set_ylim(0,0.1)
    ax[3].title.set_text('nu')
    ax[4].plot(Params_test[4][Number,15,:],'.')
    ax[4].plot(np.squeeze(p[4][Number,15,:,:]),'.')
    ax[4].set_ylim(-.1,.1)
    ax[4].title.set_text('chi_nb')
    plt.show()


    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].hist(Params_test[0][Number,:,:].ravel(),range=((0,1)))
    ax[0].title.set_text('S0')
    ax[1].hist(Params_test[1][Number,:,:].ravel(),range=((0,30)))
    ax[1].title.set_text('R2')
    ax[2].hist(Params_test[2][Number,:,:].ravel(),range=((0,1)))
    ax[2].title.set_text('Y')
    ax[3].hist(Params_test[3][Number,:,:].ravel(),range=((0,0.1)))
    ax[3].title.set_text('nu')
    ax[4].hist(Params_test[4][Number,:,:].ravel(),range=((-.1,.1)))
    ax[4].title.set_text('chi_nb')
    ax[5].hist(np.squeeze(p[0][Number,:,:,:]).ravel(),range=((0,1)))
    ax[6].hist(np.squeeze(p[1][Number,:,:,:]).ravel(),range=((0,30)))
    ax[7].hist(np.squeeze(p[2][Number,:,:,:]).ravel(),range=((0,1)))
    ax[8].hist(np.squeeze(p[3][Number,:,:,:]).ravel(),range=((0,.1)))
    ax[9].hist(np.squeeze(p[4][Number,:,:,:]).ravel(),range=((-.1,.1)))
    plt.show()

label_transformed=translate_Params(data_element[1])
prediction_transforemd=translate_Params(p)
check_Params_transformed(label_transformed,prediction_transforemd,Number)

#%% calculate MSE, histogram for whole test set like in Simons thesis
m_S0 = tf.keras.metrics.MeanRelativeError(normalizer=label_transformed[0])
m_S0.update_state(label_transformed[0],prediction_transforemd[0])
m_S0.result().numpy()

m_R2 = tf.keras.metrics.MeanRelativeError(normalizer=label_transformed[1])
m_R2.update_state(label_transformed[1],prediction_transforemd[1])
m_R2.result().numpy()

m_Y = tf.keras.metrics.MeanRelativeError(normalizer=label_transformed[2])
m_Y.update_state(label_transformed[2],prediction_transforemd[2])
m_Y.result().numpy()

m_nu = tf.keras.metrics.MeanRelativeError(normalizer=label_transformed[3])
m_nu.update_state(label_transformed[3],prediction_transforemd[3])
m_nu.result().numpy()

m_chi_nb = tf.keras.metrics.MeanRelativeError(normalizer=label_transformed[4])
m_chi_nb.update_state(label_transformed[4],prediction_transforemd[4])
m_chi_nb.result().numpy()


#%%
Number=2
QSM_test.shape
plt.figure()
plt.imshow(QSM_test[Number,:,:,0], cmap='gray')

# %%

# Second training step
"""

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
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=opt,
    metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
    #metrics=["accuracy"],
)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]
history = model.fit([qBOLD_training,QSM_training], [qBOLD_training,QSM_training] , batch_size=100, epochs=1000, validation_split=0.2, callbacks=my_callbacks)

test_scores = model.evaluate([qBOLD_test,QSM_test],  [qBOLD_test,QSM_test], verbose=2)
#qBOLD_test.shape
#qBOLD_test2 = tf.expand_dims(qBOLD_test,axis=3)
#qBOLD_test2.shape
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
test_scores_params = model_params.evaluate([qBOLD_test,QSM_test], Params_test, verbose=2)

model_params.save("models/"+version+ "Model_2D_fully_conv_Params_after_qqbold.h5")
model.save("models/"+version+ "Model_2D_fully_conv_Full_qqbold.h5")

# %%
print(history.history.keys())
plot_loss(history,'')
plot_loss(history,'qBOLD_')
plot_loss(history,'QSM_')

#%%
p_after = model_params.predict([qBOLD_test,QSM_test])
#%%
Number = 2

check_Params(Params_test,p_after,Number)

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

    P5=ax[5].imshow(p[Number,:,:,0], cmap='gray')
    #plt.colorbar(P5,ax=ax[5])
    P5.set_clim(.0,.6)

    P6=ax[6].imshow(p[Number,:,:,3], cmap='gray')
    #plt.colorbar(P6,ax=ax[6])
    P6.set_clim(.0,.6)

    P7=ax[7].imshow(p[Number,:,:,7], cmap='gray')
    #plt.colorbar(P7,ax=ax[7])
    P7.set_clim(.0,.6)

    P8=ax[8].imshow(p[Number,:,:,11], cmap='gray')
    #plt.colorbar(P8,ax=ax[8])
    P8.set_clim(.0,.6)

    P9=ax[9].imshow(p[Number,:,:,15], cmap='gray')
    plt.colorbar(P9,ax=ax[9])
    P9.set_clim(.0,.6)
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
    plt.show()
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,10,15,:],"o-r")
    ax[0].plot(t,prediction[Number,10,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,10,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,10,15,0],"ob")
    plt.show(    )
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,15,15,:],"o-r")
    ax[0].plot(t,prediction[Number,15,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,15,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,15,15,0],"ob")
    plt.show()
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,20,15,:],"o-r")
    ax[0].plot(t,prediction[Number,20,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,20,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,20,15,0],"ob")
    plt.show()
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,25,15,:],"o-r")
    ax[0].plot(t,prediction[Number,25,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,25,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,25,15,0],"ob")
    plt.show()

Number=2
check_Pixel(qBOLD_test,p_full[0],QSM_test,p_full[1],Number)
"""
