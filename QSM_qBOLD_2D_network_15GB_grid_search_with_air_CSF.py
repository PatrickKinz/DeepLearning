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

Dataset_train=np.load("../Brain_Phantom/Patches_with_air_big/15GB_1Pnoise_train_val.npz")
S0_train=Dataset_train['S0']
S0_train.shape
R2_train=Dataset_train['R2']
Y_train=Dataset_train['Y']
nu_train=Dataset_train['nu']
chi_nb_train=Dataset_train['chi_nb']
qBOLD_training=Dataset_train['qBOLD']
QSM_training=Dataset_train['QSM']
Seg_training=Dataset_train['Seg']

training_list = [S0_train,R2_train,Y_train,nu_train,chi_nb_train]
#%%
Dataset_test=np.load("../Brain_Phantom/Patches_with_air_big/15GB_1Pnoise_test.npz")
S0_test=Dataset_test['S0']
S0_test.shape
R2_test=Dataset_test['R2']
Y_test=Dataset_test['Y']
nu_test=Dataset_test['nu']
chi_nb_test=Dataset_test['chi_nb']
qBOLD_test=Dataset_test['qBOLD']
QSM_test=Dataset_test['QSM']
Seg_test=Dataset_test['Seg']

test_list = [S0_test,R2_test,Y_test,nu_test,chi_nb_test]


version = "with_air_1Pnoise_15GB/"

# %% Look at data
def check_data(qBOLD,QSM,S0,R2,Y,nu,chi_nb,Seg,Number): #target prediction
    fig, axes = plt.subplots(nrows=2, ncols=6,figsize=(15,5))
    ax = axes.ravel()

    P0=ax[0].imshow(qBOLD[Number,:,:,0], cmap='inferno')
    ax[0].title.set_text('3ms')
    #plt.colorbar(P0,ax=ax[0])
    P0.set_clim(.0,1)

    P1=ax[1].imshow(qBOLD[Number,:,:,3], cmap='inferno')
    ax[1].title.set_text('9ms')
    #plt.colorbar(P1,ax=ax[1])
    P1.set_clim(.0,1)

    P2=ax[2].imshow(qBOLD[Number,:,:,7], cmap='inferno')
    ax[2].title.set_text('21ms')
    #plt.colorbar(P2,ax=ax[2])
    P2.set_clim(.0,1)

    P3=ax[3].imshow(qBOLD[Number,:,:,11], cmap='inferno')
    ax[3].title.set_text('33ms')
    #plt.colorbar(P3,ax=ax[3])
    P3.set_clim(.0,1)

    P4=ax[4].imshow(qBOLD[Number,:,:,15], cmap='inferno')
    ax[4].title.set_text('45ms')
    #plt.colorbar(P4,ax=ax[4])
    P4.set_clim(.0,1)

    P5=ax[5].imshow(QSM[Number,:,:], cmap='inferno')
    ax[5].title.set_text('QSM')
    #plt.colorbar(P5,ax=ax[5])
    P5.set_clim(-.11,+0.11)

    P6=ax[6].imshow(S0[Number,:,:], cmap='inferno')
    ax[6].title.set_text('a')
    #plt.colorbar(P6,ax=ax[6])
    P6.set_clim(0,1)

    P7=ax[7].imshow(R2[Number,:,:], cmap='inferno')
    ax[7].title.set_text('b')
    #plt.colorbar(P7,ax=ax[7])
    P7.set_clim(0,1)

    P8=ax[8].imshow(Y[Number,:,:], cmap='inferno')
    ax[8].title.set_text('c')
    #plt.colorbar(P8,ax=ax[8])
    P8.set_clim(0,1)

    P9=ax[9].imshow(nu[Number,:,:], cmap='inferno')
    ax[9].title.set_text('d')
    #plt.colorbar(P9,ax=ax[9])
    P9.set_clim(0,1)

    P10=ax[10].imshow(chi_nb[Number,:,:], cmap='inferno')
    ax[10].title.set_text('e')
    #plt.colorbar(P10,ax=ax[10])
    P10.set_clim(0,1)

    P11=ax[11].imshow(Seg[Number,:,:], cmap='inferno')
    ax[11].title.set_text('Seg')
    #plt.colorbar(P11,ax=ax[11])
    P11.set_clim(0,2)
    plt.show()

check_data(qBOLD_test,QSM_test,S0_test,R2_test,Y_test,nu_test,chi_nb_test,Seg_test,1)
check_data(qBOLD_test,QSM_test,S0_test,R2_test,Y_test,nu_test,chi_nb_test,Seg_test,2)
check_data(qBOLD_test,QSM_test,S0_test,R2_test,Y_test,nu_test,chi_nb_test,Seg_test,3)







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
                  activation='tanh',
                  name='conv_qBOLD_1')(input_qBOLD)

conv_qBOLD_2 = keras.layers.Conv2D(2*n,
                  kernel_size = 3,
                  strides=1,
                  padding='same',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_qBOLD_2')(conv_qBOLD_1)

conv_qBOLD_3 = keras.layers.Conv2D(3*n,
                  kernel_size = 3,
                  strides=1,
                  padding='same',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_qBOLD_3')(conv_qBOLD_2)

model_qBOLD = keras.Model(inputs=input_qBOLD, outputs = conv_qBOLD_3, name="qBOLD model")
model_qBOLD.summary()
keras.utils.plot_model(model_qBOLD, show_shapes=True)


input_QSM = keras.Input(shape=(30,30,1), name = 'Input_QSM')
conv_QSM_1 = keras.layers.Conv2D(n,
                  kernel_size=3,
                  strides=(1),
                  padding='same',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_QSM_1')(input_QSM)

conv_QSM_2 = keras.layers.Conv2D(2*n,
                  kernel_size=3,
                  strides=(1),
                  padding='same',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_QSM_2')(conv_QSM_1)


model_QSM = keras.Model(inputs=input_QSM, outputs = conv_QSM_2, name="QSM model")
model_QSM.summary()
keras.utils.plot_model(model_QSM, show_shapes=True)

#%% Network Segmentation
conv_Seg_1 = keras.layers.Conv2D(8,
                kernel_size=3,
                strides=(1),
                padding='same',
                dilation_rate=1,
                activation='tanh',
                name='conv_Seg_1')(input_qBOLD)
conv_Seg_2 = keras.layers.Conv2D(16,
                kernel_size=3,
                strides=(1),
                padding='same',
                dilation_rate=1,
                activation='tanh',
                name='conv_Seg_2')(conv_Seg_1)
conv_Seg_3 = keras.layers.Conv2D(32,
                kernel_size=3,
                strides=(1),
                padding='same',
                dilation_rate=1,
                activation='tanh',
                name='conv_Seg_3')(conv_Seg_2)
n_types = 3
conv_Seg_4 = keras.layers.Conv2D(n_types,
                kernel_size=3,
                strides=(1),
                padding='same',
                dilation_rate=1,
                activation='relu',
                name='conv_Seg_4')(conv_Seg_3)

model_Seg = keras.Model(inputs=input_qBOLD, outputs = conv_Seg_4, name="Seg_model")
model_Seg.summary()
keras.utils.plot_model(model_Seg, show_shapes=True)

opt = keras.optimizers.Adam(0.001, clipnorm=1.)
model_Seg.compile(optimizer=opt,loss=keras.losses.SparseCategoricalCrossentropy())

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]

#%%
history_Seg = model_Seg.fit(qBOLD_training, Seg_training , batch_size=100, epochs=100, validation_split=0.1/0.9, callbacks=my_callbacks)


#%%
p_Seg = model_Seg.predict(qBOLD_test)
p_Seg[0].shape

def check_Seg(Seg_true,Seg_pred,Number): #target prediction
    fig, axes = plt.subplots(nrows=1, ncols=5,figsize=(15,5))
    ax = axes.ravel()

    P0=ax[0].imshow(Seg_true[Number,:,:], cmap='inferno')
    ax[0].title.set_text('true')
    #plt.colorbar(P0,ax=ax[0])
    P0.set_clim(.0,2)

    P1=ax[1].imshow(Seg_pred[Number,:,:,0], cmap='inferno')
    ax[1].title.set_text('tissue')
    #plt.colorbar(P1,ax=ax[1])
    P1.set_clim(.0,1)

    P2=ax[2].imshow(Seg_pred[Number,:,:,1], cmap='inferno')
    ax[2].title.set_text('air')
    #plt.colorbar(P2,ax=ax[2])
    P2.set_clim(.0,1)

    P3=ax[3].imshow(Seg_pred[Number,:,:,2], cmap='inferno')
    ax[3].title.set_text('CSF')
    #plt.colorbar(P3,ax=ax[3])
    P3.set_clim(.0,1)

    seg_pred = tf.math.argmax(Seg_pred[Number,:,:,:], axis=-1)
    P4=ax[4].imshow(seg_pred, cmap='inferno')
    ax[4].title.set_text('Pred')
    #plt.colorbar(P3,ax=ax[3])
    P4.set_clim(.0,2)
    plt.show()

check_Seg(Seg_test,p_Seg,1)
check_Seg(Seg_test,p_Seg,2)
check_Seg(Seg_test,p_Seg,3)
check_Seg(Seg_test,p_Seg,9)
check_Seg(Seg_test,p_Seg,19)
#%%

concat_QQ_1 = layers.Concatenate(name = 'concat_QQ_1')([model_qBOLD.output,model_QSM.output,model_Seg.output])
conv_QQ_1 = layers.Conv2D(2*n,3,padding='same',activation="tanh",name = 'conv_QQ_1')(concat_QQ_1)
#conv_QQ_1 = layers.Conv2D(2*n,3,padding='same',activation="tanh",name = 'conv_QQ_1')(model_qBOLD.output)

conv_QQ_2 = layers.Conv2D(4*n,3,padding='same',activation="tanh",name = 'conv_QQ_2')(conv_QQ_1)
conv_QQ_3 = layers.Conv2D(8*n,3,padding='same',activation="tanh",name = 'conv_QQ_3')(conv_QQ_2)


conv_S0 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'S0')(    conv_QQ_3)
conv_R2 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'R2')(    conv_QQ_3)
conv_Y = layers.Conv2D(1,3,padding='same',activation="linear", name = 'Y')(     conv_QQ_3)
conv_nu = layers.Conv2D(1,3,padding='same',activation="linear", name = 'nu')(    conv_QQ_3)
conv_chinb = layers.Conv2D(1,3,padding='same',activation="linear", name = 'chi_nb')(conv_QQ_3)


model_params = keras.Model(inputs=[input_qBOLD,input_QSM],outputs=[conv_S0,conv_R2,conv_Y,conv_nu,conv_chinb],name="Params_model")
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
}
lossWeights = {
    "S0":1.0,
    "R2":1.0,
    "Y":1.0,
    "nu":1.0,
    "chi_nb":1.0,
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

for l in model_Seg.layers:
    l.trainable=False



history_params = model_params.fit([qBOLD_training,QSM_training], training_list , batch_size=100, epochs=100, validation_split=0.1/0.9, callbacks=my_callbacks)
#history_params = model_params.fit(training_Params_data, epochs=100,validation_data=val_Params_data, callbacks=my_callbacks)
#%%
#model_params.save("models/"+version+ "Model_2D_fully_conv_Params_deeper_with_air.h5")
#np.save('models/'+version+'history_params_2D_fully_conv_Params_deeper_with_air.h5.npy',history_params.history)
model_params = keras.models.load_model("models/"+version+ "Model_2D_fully_conv_Params_deeper_with_air.h5")
model_params.summary()
keras.utils.plot_model(model_params, show_shapes=True)

#%%


test_scores = model_params.evaluate([qBOLD_test,QSM_test], test_list, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
#%%
print(history_params.history.keys())

QQplt.plot_loss(history_params,'')
QQplt.plot_loss(history_params,'S0_')
QQplt.plot_loss(history_params,'Y_')
QQplt.plot_loss(history_params,'nu_')
QQplt.plot_loss(history_params,'chi_nb_')


# %%
#model_params = keras.models.load_model("models/"+version+ "Model_2D_Params_before_qqbold.h5")
#model_params.summary()
p = model_params.predict([qBOLD_test,QSM_test])
p[0].shape

#%%
Number=1
label_transformed=QQplt.remove_air_and_CSF(QQplt.translate_Params(test_list),Seg_test)
prediction_transformed=QQplt.remove_air_and_CSF(QQplt.translate_Params(p),Seg_test)
label_transformed[0].shape
prediction_transformed[0].shape
QQplt.check_full_confusion_matrix(label_transformed,prediction_transformed,'confusion_test')

#%%
#label_transformed_array =np.array(label_transformed)
#label_transformed_array.shape
#prediction_transformed_array = np.array(prediction_transformed)
#Cov_array = np.corrcoef(label_transformed,prediction_transformed)
Cov_array=np.zeros((5,5))
#print(Cov_array)
def correlation_coef(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    SPxy = np.sum((x - x_mean)*(y -y_mean))
    SQx = np.sum((x-x_mean)*(x-x_mean))
    SQy = np.sum((y-y_mean)*(y-y_mean))
    return SPxy/np.sqrt(SQx*SQy)

for i in range(5):
    for j in range(5):
        Cov_array[i,j] = correlation_coef(label_transformed[j],prediction_transformed[i])
Cov_array_round = np.round(Cov_array,3)
print(Cov_array_round)
#%%
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(7,6))
P0 = ax.imshow(Cov_array, cmap='bwr')
P0.set_clim(-1,1)
ax.title.set_text('Pearson correlation coefficient')
plt.colorbar(P0,ax=ax)
ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels(['S$_0$ true','R$_2$ true','Y true','$v$ true','$\chi_{nb}$ true'])
ax.set_yticks([0,1,2,3,4])
ax.set_yticklabels(['S$_0$ pred','R$_2$ pred','Y pred','$v$ pred','$\chi_{nb}$ pred'])
for i in range(5):
    for j in range(5):
        c = Cov_array_round[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
fig.savefig('plots/'+'confusion_test_correlation_coeffs'+'.png')



#%%
QQplt.check_Params_transformed(label_transformed,prediction_transformed,Number,'CNN_Uniform_GESFIDE_16Echoes_Params_with_air')

QQplt.check_Params_transformed_hist(label_transformed,prediction_transformed,'CNN_Uniform_GESFIDE_16Echoes_evaluation_with_air')

QQplt.check_nu_calc(label_transformed,prediction_transformed,QSM_test)
def check_nu_calc(Params_test,p,QSM_test):
    nu_calc = QQfunc.f_nu(p[2],p[4],QSM_test)

    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(5,5))
    counts, xedges, yedges, im = axes.hist2d(x=Params_test[3][:,:,:].ravel()*100,y=np.squeeze(nu_calc[:,:,:,:]).ravel()*100,bins=30,range=((1,10),(-5,15)),cmap='inferno')
    axes.title.set_text('$v$ [%]')
    axes.set_xlabel('truth')
    axes.set_ylabel('calculation')
    cbar=fig.colorbar(im,ax=axes)
    cbar.formatter.set_powerlimits((0, 0))
    axes.plot(np.linspace(0,10,10),np.linspace(0,10,10))
    plt.show()

check_nu_calc(label_transformed,prediction_transformed,QSM_test)


QQplt.check_nu_calc_QSM_noiseless(label_transformed,prediction_transformed,test_list)


nu_calc = QQfunc.f_nu(prediction_transformed[2],prediction_transformed[4],QSM_test)

def check_Params_transformed_hist(Params_test,p,QSM_test,filename):
    nu_calc = QQfunc.f_nu(p[2],p[4],QSM_test)

    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].hist(Params_test[0][:,:,:].ravel(),range=((0,1)))
    ax[0].title.set_text('S0')
    ax[1].hist(Params_test[1][:,:,:].ravel(),range=((0,30)))
    ax[1].title.set_text('R2')
    ax[2].hist(Params_test[2][:,:,:].ravel(),range=((0,1)))
    ax[2].title.set_text('Y')
    ax[3].hist(Params_test[3][:,:,:].ravel(),range=((0,0.1)))
    ax[3].title.set_text('nu')
    ax[4].hist(Params_test[4][:,:,:].ravel(),range=((-.1,.1)))
    ax[4].title.set_text('chi_nb')
    ax[5].hist(np.squeeze(p[0][:,:,:,:]).ravel(),range=((0,1)))
    ax[6].hist(np.squeeze(p[1][:,:,:,:]).ravel(),range=((0,30)))
    ax[7].hist(np.squeeze(p[2][:,:,:,:]).ravel(),range=((0,1)))
    ax[8].hist(np.squeeze(p[3][:,:,:,:]).ravel(),range=((0,.1)))
    ax[9].hist(np.squeeze(p[4][:,:,:,:]).ravel(),range=((-.1,.1)))
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(12,7))
    ax = axes.ravel()
    counts, xedges, yedges, im = ax[0].hist2d(x=Params_test[0][:,:,:].ravel(),y=np.squeeze(p[0][:,:,:,:]).ravel(),bins=30,range=((0.1,1),(0.1,1)),cmap='inferno')
    ax[0].title.set_text('$S_0$ [a.u.]')
    ax[0].set_xlabel('truth')
    ax[0].set_ylabel('prediction')
    cbar=fig.colorbar(im,ax=ax[0])
    cbar.formatter.set_powerlimits((0, 0))
    counts, xedges, yedges, im = ax[1].hist2d(x=Params_test[1][:,:,:].ravel(),y=np.squeeze(p[1][:,:,:,:]).ravel(),bins=30,range=((5,30),(5,30)),cmap='inferno')
    ax[1].title.set_text('$R_2$ [Hz]')
    ax[1].set_xlabel('truth')
    ax[1].set_ylabel('prediction')
    cbar=fig.colorbar(im,ax=ax[1])
    cbar.formatter.set_powerlimits((0, 0))
    counts_Y, xedges_Y, yedges, im = ax[2].hist2d(x=Params_test[2][:,:,:].ravel()*100,y=np.squeeze(p[2][:,:,:,:]).ravel()*100,bins=30,range=((5,98),(5,98)),cmap='inferno')
    ax[2].title.set_text('Y [%]')
    ax[2].set_xlabel('truth')
    ax[2].set_ylabel('prediction')
    ax[2].plot(np.linspace(5,98,10),np.linspace(5,98,10))
    cbar=fig.colorbar(im,ax=ax[2])
    cbar.formatter.set_powerlimits((0, 0))
    counts, xedges, yedges, im = ax[3].hist2d(x=Params_test[3][:,:,:].ravel()*100,y=np.squeeze(p[3][:,:,:,:]).ravel()*100,bins=30,range=((1,.1*100),(1,.1*100)),cmap='inferno')
    ax[3].title.set_text('$v$ [%]')
    ax[3].set_xlabel('truth')
    ax[3].set_ylabel('prediction')
    ax[3].plot(np.linspace(1,10,10),np.linspace(1,10,10))
    cbar=fig.colorbar(im,ax=ax[3])
    cbar.formatter.set_powerlimits((0, 0))
    counts, xedges, yedges, im = ax[4].hist2d(x=Params_test[3][:,:,:].ravel()*100,y=np.squeeze(nu_calc[:,:,:,:]).ravel()*100,bins=30,range=((1,10),(1,10)),cmap='inferno')
    ax[4].title.set_text('$v$ calc [%]')
    ax[4].set_xlabel('truth')
    ax[4].set_ylabel('prediction')
    ax[4].plot(np.linspace(1,10,10),np.linspace(1,10,10))
    cbar=fig.colorbar(im,ax=ax[4])
    cbar.formatter.set_powerlimits((0, 0))
    counts, xedges, yedges, im = ax[5].hist2d(x=Params_test[4][:,:,:].ravel()*1000,y=np.squeeze(p[4][:,:,:,:]).ravel()*1000,bins=30,range=((-100+10,100),(-100+10,100)),cmap='inferno')
    ax[5].title.set_text('$\chi_{nb}$ [ppb]')
    ax[5].set_xlabel('truth')
    ax[5].set_ylabel('prediction')
    cbar=fig.colorbar(im,ax=ax[5])
    cbar.formatter.set_powerlimits((0, 0))
    plt.tight_layout()
    plt.show()
    fig.savefig('plots/'+filename+'.png')

check_Params_transformed_hist(label_transformed,prediction_transformed,QSM_test,'CNN_Uniform_GESFIDE_16Echoes_evaluation_with_air')


#somehow combine v CNN and v calc
"""
if v_CNN > 4.5 :
    v = v_CNN
else:
    if v_true < 4 :
        v = v_CNN
    else:
        v = v_calc
"""
"""
if v_calc > 4.5 and v_CNN > 4
    v = v_CNN
if v_calc > 4.5 and v_CNN < 4
    v = v_calc
if v_calc > 4.5
    v = c_CNN
"""
#%%
def check_nu_CNN_nu_calc(Params_test,p,QSM_test):
    nu_calc = QQfunc.f_nu(p[2],p[4],QSM_test).ravel()*100
    nu_CNN = p[3][:,:,:].ravel()*100
    nu = np.zeros_like(nu_calc)
    nu_comb2 = np.zeros_like(nu_calc)
    for i in tqdm(range(len(nu_calc))):
        if nu_calc[i] > 4:
            if nu_CNN[i] > 4:
                nu[i] = nu_CNN[i]
            else:
                nu[i] = nu_calc[i]
        else:
            nu[i] = nu_CNN[i]
        if nu_calc[i]-nu_CNN[i] > 1:
            nu_comb2[i] = nu_calc[i]
        else:
            nu_comb2[i] = nu_CNN[i]



    fig, ax = plt.subplots(nrows=2, ncols=3,figsize=(20,10))
    axes = ax.ravel()
    counts, xedges, yedges, im = axes[0].hist2d(x=Params_test[3][:,:,:].ravel()*100,y=nu_CNN,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[0].title.set_text('$v$ [%]')
    axes[0].set_xlabel('truth')
    axes[0].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[0])
    cbar.formatter.set_powerlimits((0, 0))
    axes[0].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[1].hist2d(x=Params_test[3][:,:,:].ravel()*100,y=nu_calc,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[1].title.set_text('$v$ [%]')
    axes[1].set_xlabel('truth')
    axes[1].set_ylabel('calculation')
    cbar=fig.colorbar(im,ax=axes[1])
    cbar.formatter.set_powerlimits((0, 0))
    axes[1].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[2].hist2d(x=Params_test[3][:,:,:].ravel()*100,y=nu,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[2].title.set_text('$v$ [%]')
    axes[2].set_xlabel('truth')
    axes[2].set_ylabel('combined')
    cbar=fig.colorbar(im,ax=axes[2])
    cbar.formatter.set_powerlimits((0, 0))
    axes[2].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[3].hist2d(x=nu_calc,y=nu_CNN,bins=30,range=((1,10),(2,10)),cmap='inferno')
    axes[3].title.set_text('$v$ [%]')
    axes[3].set_xlabel('calc')
    axes[3].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[3])
    cbar.formatter.set_powerlimits((0, 0))
    axes[3].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[4].hist2d(x=Params_test[3][:,:,:].ravel()*100,y=nu_calc - nu_CNN,bins=30,range=((1,10),(-2.5,10)),cmap='inferno')
    axes[4].title.set_text('$v$ [%]')
    axes[4].set_xlabel('truth')
    axes[4].set_ylabel('calc - CNN')
    cbar=fig.colorbar(im,ax=axes[4])
    cbar.formatter.set_powerlimits((0, 0))
    axes[4].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[5].hist2d(x=Params_test[3][:,:,:].ravel()*100,y=nu_comb2,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[5].title.set_text('$v$ [%]')
    axes[5].set_xlabel('truth')
    axes[5].set_ylabel('combined 2')
    cbar=fig.colorbar(im,ax=axes[5])
    cbar.formatter.set_powerlimits((0, 0))
    axes[5].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    plt.show()

check_nu_CNN_nu_calc(label_transformed,prediction_transformed,QSM_test)

p[2].ravel().shape
#%%
def check_nu_CNN_compared_to_OEF(Params_test,p,QSM_test):
    nu_CNN = p[3].ravel()*100
    nu_CNN_1 = []
    nu_CNN_2 = []
    nu_CNN_3 = []
    nu_CNN_4 = []
    nu_CNN_5 = []
    nu_true = Params_test[3].ravel()*100
    nu_true_1 = []
    nu_true_2 = []
    nu_true_3 = []
    nu_true_4 = []
    nu_true_5 = []
    OEF_true = Params_test[2].ravel()
    for i in  tqdm(range(len(OEF_true))):
        if OEF_true[i] > 0.90:
            nu_CNN_5.append(nu_CNN[i])
            nu_true_5.append(nu_true[i])
        elif OEF_true[i] <= 0.90 and OEF_true[i] > 0.80:
            nu_CNN_4.append(nu_CNN[i])
            nu_true_4.append(nu_true[i])
        elif OEF_true[i] <= 0.80 and OEF_true[i] > 0.70:
            nu_CNN_3.append(nu_CNN[i])
            nu_true_3.append(nu_true[i])
        elif OEF_true[i] <= 0.70 and OEF_true[i] > 0.60:
            nu_CNN_2.append(nu_CNN[i])
            nu_true_2.append(nu_true[i])
        else:
            nu_CNN_1.append(nu_CNN[i])
            nu_true_1.append(nu_true[i])

    fig, ax = plt.subplots(nrows=2, ncols=3,figsize=(20,10))
    axes = ax.ravel()
    counts, xedges, yedges, im = axes[0].hist2d(x=nu_true,y=nu_CNN,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[0].title.set_text('$v$ [%] all')
    axes[0].set_xlabel('truth')
    axes[0].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[0])
    cbar.formatter.set_powerlimits((0, 0))
    axes[0].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[1].hist2d(x=nu_true_1,y=nu_CNN_1,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[1].title.set_text('$v$ [%] OEF < 60 %')
    axes[1].set_xlabel('truth')
    axes[1].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[1])
    cbar.formatter.set_powerlimits((0, 0))
    axes[1].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[2].hist2d(x=nu_true_2,y=nu_CNN_2,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[2].title.set_text('$v$ [%] 60 % < OEF < 70 %')
    axes[2].set_xlabel('truth')
    axes[2].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[2])
    cbar.formatter.set_powerlimits((0, 0))
    axes[2].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[3].hist2d(x=nu_true_3,y=nu_CNN_3,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[3].title.set_text('$v$ [%] 70 % < OEF < 80 %')
    axes[3].set_xlabel('truth')
    axes[3].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[3])
    cbar.formatter.set_powerlimits((0, 0))
    axes[3].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[4].hist2d(x=nu_true_4,y=nu_CNN_4,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[4].title.set_text('$v$ [%] 80 % < OEF < 90 %')
    axes[4].set_xlabel('truth')
    axes[4].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[4])
    cbar.formatter.set_powerlimits((0, 0))
    axes[4].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[5].hist2d(x=nu_true_5,y=nu_CNN_5,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[5].title.set_text('$v$ [%] 90 % < OEF')
    axes[5].set_xlabel('truth')
    axes[5].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[5])
    cbar.formatter.set_powerlimits((0, 0))
    axes[5].plot(np.linspace(0,10,10),np.linspace(0,10,10))

check_nu_CNN_compared_to_OEF(label_transformed,prediction_transformed,QSM_test)

#%%
def check_nu_calc_compared_to_OEF(Params_test,p,QSM_test):
    nu_CNN = QQfunc.f_nu(p[2],p[4],QSM_test).ravel()*100
    nu_CNN_1 = []
    nu_CNN_2 = []
    nu_CNN_3 = []
    nu_CNN_4 = []
    nu_CNN_5 = []
    nu_true = Params_test[3].ravel()*100
    nu_true_1 = []
    nu_true_2 = []
    nu_true_3 = []
    nu_true_4 = []
    nu_true_5 = []
    OEF_true = Params_test[2].ravel()
    for i in  tqdm(range(len(OEF_true))):
        if OEF_true[i] > 0.90:
            nu_CNN_5.append(nu_CNN[i])
            nu_true_5.append(nu_true[i])
        elif OEF_true[i] <= 0.90 and OEF_true[i] > 0.80:
            nu_CNN_4.append(nu_CNN[i])
            nu_true_4.append(nu_true[i])
        elif OEF_true[i] <= 0.80 and OEF_true[i] > 0.70:
            nu_CNN_3.append(nu_CNN[i])
            nu_true_3.append(nu_true[i])
        elif OEF_true[i] <= 0.70 and OEF_true[i] > 0.60:
            nu_CNN_2.append(nu_CNN[i])
            nu_true_2.append(nu_true[i])
        else:
            nu_CNN_1.append(nu_CNN[i])
            nu_true_1.append(nu_true[i])

    fig, ax = plt.subplots(nrows=2, ncols=3,figsize=(20,10))
    axes = ax.ravel()
    counts, xedges, yedges, im = axes[0].hist2d(x=nu_true,y=nu_CNN,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[0].title.set_text('$v$ [%]')
    axes[0].set_xlabel('truth')
    axes[0].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[0])
    cbar.formatter.set_powerlimits((0, 0))
    axes[0].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[1].hist2d(x=nu_true_1,y=nu_CNN_1,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[1].title.set_text('$v$ [%] OEF < 60 %')
    axes[1].set_xlabel('truth')
    axes[1].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[1])
    cbar.formatter.set_powerlimits((0, 0))
    axes[1].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[2].hist2d(x=nu_true_2,y=nu_CNN_2,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[2].title.set_text('$v$ [%] 60 % < OEF < 70 %')
    axes[2].set_xlabel('truth')
    axes[2].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[2])
    cbar.formatter.set_powerlimits((0, 0))
    axes[2].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[3].hist2d(x=nu_true_3,y=nu_CNN_3,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[3].title.set_text('$v$ [%] 70 % < OEF < 80 %')
    axes[3].set_xlabel('truth')
    axes[3].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[3])
    cbar.formatter.set_powerlimits((0, 0))
    axes[3].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[4].hist2d(x=nu_true_4,y=nu_CNN_4,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[4].title.set_text('$v$ [%] 80 % < OEF < 90 %')
    axes[4].set_xlabel('truth')
    axes[4].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[4])
    cbar.formatter.set_powerlimits((0, 0))
    axes[4].plot(np.linspace(0,10,10),np.linspace(0,10,10))
    counts, xedges, yedges, im = axes[5].hist2d(x=nu_true_5,y=nu_CNN_5,bins=30,range=((1,10),(0,10)),cmap='inferno')
    axes[5].title.set_text('$v$ [%] 90 % < OEF')
    axes[5].set_xlabel('truth')
    axes[5].set_ylabel('CNN')
    cbar=fig.colorbar(im,ax=axes[5])
    cbar.formatter.set_powerlimits((0, 0))
    axes[5].plot(np.linspace(0,10,10),np.linspace(0,10,10))

check_nu_calc_compared_to_OEF(label_transformed,prediction_transformed,QSM_test)

#%%
calc_nu_explicit = layers.Lambda(QQfunc.f_nu_tensor_2, name="nu_calc")([model_params.output[3],model_params.output[4],model_params.input[1]])

concat_nu_calc_1 = layers.Concatenate(name = 'concat_nu_calc_1')([model_qBOLD.input,model_QSM.input,model_params.output[0],model_params.output[1],model_params.output[2],model_params.output[3],model_params.output[4],calc_nu_explicit])
conv_nu_calc_1 = layers.Conv2D(2*n,3,padding='same',activation="tanh",name = 'conv_nu_calc_1')(concat_nu_calc_1)
#conv_QQ_1 = layers.Conv2D(2*n,3,padding='same',activation="tanh",name = 'conv_QQ_1')(model_qBOLD.output)

conv_nu_calc_2 = layers.Conv2D(4*n,3,padding='same',activation="tanh",name = 'conv_nu_calc_2')(conv_nu_calc_1)
conv_nu_calc_3 = layers.Conv2D(8*n,3,padding='same',activation="tanh",name = 'conv_nu_calc_3')(conv_nu_calc_2)


conv_S0_2 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'S0_2')(    conv_nu_calc_3)
conv_R2_2 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'R2_2')(    conv_nu_calc_3)
conv_Y_2 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'Y_2')(     conv_nu_calc_3)
conv_nu_2 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'nu_2')(    conv_nu_calc_3)
conv_chinb_2 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'chi_nb_2')(conv_nu_calc_3)


model_params_nu_calc = keras.Model(inputs=[input_qBOLD,input_QSM],outputs=[conv_S0_2,conv_R2_2,conv_Y_2,conv_nu_2,conv_chinb_2],name="Params_model_nu_calc")
model_params_nu_calc.summary()
keras.utils.plot_model(model_params_nu_calc, show_shapes=True)

# %% Train Params model

opt = keras.optimizers.Adam(0.001, clipnorm=1.)
#loss=keras.losses.MeanAbsolutePercentageError()
#loss=keras.losses.MeanSquaredLogarithmicError()
loss=keras.losses.MeanAbsoluteError()
#loss=tf.keras.losses.Huber()
losses = {
    "S0_2":loss,
    "R2_2":loss,
    "Y_2":loss,
    "nu_2":loss,
    "chi_nb_2":loss,
}
lossWeights = {
    "S0_2":1.0,
    "R2_2":1.0,
    "Y_2":1.0,
    "nu_2":1.0,
    "chi_nb_2":1.0,
}
model_params_nu_calc.compile(
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

history_params = model_params_nu_calc.fit([qBOLD_training,QSM_training], training_list , batch_size=100, epochs=100, validation_split=0.1/0.9, callbacks=my_callbacks)



#%%
p2 = model_params_nu_calc.predict([qBOLD_test,QSM_test])
p2[0].shape

#%%
prediction_transformed2=QQplt.translate_Params(p2)
check_Params_transformed_hist(label_transformed,prediction_transformed2,QSM_test,'CNN_Uniform_GESFIDE_16Echoes_evaluation_with_air_deeper_nu_calc')

#%%
fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(15,10))
ax=axes.ravel()
#y=np.maximum(np.squeeze(prediction_transformed[3][:,:,:,:]).ravel()*100,np.squeeze(nu_calc[:,:,:,:]).ravel()*100)
y0=np.squeeze(prediction_transformed[3][:,:,:,:]).ravel()*100
counts, xedges, yedges, im = ax[0].hist2d(x=label_transformed[3][:,:,:].ravel()*100,y=y0,bins=30,range=((0,10),(-5,15)),cmap='inferno')
ax[0].title.set_text('$v$ [%]')
ax[0].set_xlabel('truth')
ax[0].set_ylabel('prediction')
cbar=fig.colorbar(im,ax=ax[0])
cbar.formatter.set_powerlimits((0, 0))
ax[0].plot(np.linspace(0,10,10),np.linspace(0,10,10))
y1=np.squeeze(nu_calc[:,:,:,:]).ravel()*100
counts, xedges, yedges, im = ax[1].hist2d(x=label_transformed[3][:,:,:].ravel()*100,y=y1,bins=30,range=((0,10),(-5,15)),cmap='inferno')
ax[1].title.set_text('$v$ [%]')
ax[1].set_xlabel('truth')
ax[1].set_ylabel('calculation')
cbar=fig.colorbar(im,ax=ax[1])
cbar.formatter.set_powerlimits((0, 0))
ax[1].plot(np.linspace(0,10,10),np.linspace(0,10,10))
#y1=np.squeeze(nu_calc[:,:,:,:]).ravel()*100
counts, xedges, yedges, im = ax[2].hist2d(x=label_transformed[3][:,:,:].ravel()*100,y=np.maximum(y0,y1),bins=30,range=((0,10),(-5,15)),cmap='inferno')
ax[2].title.set_text('$v$ [%]')
ax[2].set_xlabel('truth')
ax[2].set_ylabel('max(pred,calc)')
cbar=fig.colorbar(im,ax=ax[2])
cbar.formatter.set_powerlimits((0, 0))
ax[2].plot(np.linspace(0,10,10),np.linspace(0,10,10))
#y1=np.squeeze(nu_calc[:,:,:,:]).ravel()*100
counts, xedges, yedges, im = ax[3].hist2d(x=label_transformed[3][:,:,:].ravel()*100,y=(y0+y1)/2.,bins=30,range=((0,10),(-5,15)),cmap='inferno')
ax[3].title.set_text('$v$ [%]')
ax[3].set_xlabel('truth')
ax[3].set_ylabel('(pred+calc)/2')
cbar=fig.colorbar(im,ax=ax[3])
cbar.formatter.set_powerlimits((0, 0))
ax[3].plot(np.linspace(0,10,10),np.linspace(0,10,10))
#y1=np.squeeze(nu_calc[:,:,:,:]).ravel()*100
counts, xedges, yedges, im = ax[4].hist2d(x=label_transformed[3][:,:,:].ravel()*100,y=np.minimum(y0,y1),bins=30,range=((0,10),(-5,15)),cmap='inferno')
ax[4].title.set_text('$v$ [%]')
ax[4].set_xlabel('truth')
ax[4].set_ylabel('min(pred,calc)')
cbar=fig.colorbar(im,ax=ax[4])
cbar.formatter.set_powerlimits((0, 0))
ax[4].plot(np.linspace(0,10,10),np.linspace(0,10,10))

plt.show()

#%%
Number=2
QSM_test.shape
plt.figure()
plt.imshow(QSM_test[Number,:,:,0], cmap='gray')

#%% Grid grid_search step
#input_qBOLD_2 = keras.Input(shape=(30,30,16), name = 'Input_qBOLD_2')
#input_QSM_2 = keras.Input(shape=(30,30,1), name = 'Input_QSM_2')

output_Params= model_params(inputs=[model_params.input[0],model_params.input[1]],training=False)

flat_S0 = layers.Flatten(name = 'flat_S0')(output_Params[0])
flat_R2 = layers.Flatten(name = 'flat_R2')(output_Params[1])
flat_Y = layers.Flatten(name = 'flat_Y')(output_Params[2])
flat_nu = layers.Flatten(name = 'flat_nu')(output_Params[3])
flat_chinb = layers.Flatten(name = 'flat_chinb')(output_Params[4])
flat_qBOLD = layers.Reshape((-1,16),name = 'flat_qBOLD')(model_params.input[0])
flat_QSM = layers.Flatten(name = 'flat_QSM')(model_params.input[1])

flat_S0 = layers.ReLU(max_value=1.0)(flat_S0)
flat_R2 = layers.ReLU(max_value=1.0)(flat_R2)
flat_Y = layers.ReLU(max_value=1.0)(flat_Y)
flat_nu = layers.ReLU(max_value=1.0)(flat_nu)
flat_chinb = layers.ReLU(max_value=1.0)(flat_chinb)
flat_qBOLD = layers.ReLU(max_value=1.0)(flat_qBOLD)

#flat_S0=layers.Activation('linear', dtype='float32')(flat_S0)
#flat_R2=layers.Activation('linear', dtype='float32')(flat_R2)
#flat_Y=layers.Activation('linear', dtype='float32')(flat_Y)
#flat_nu=layers.Activation('linear', dtype='float32')(flat_nu)
#flat_chinb=layers.Activation('linear', dtype='float32')(flat_chinb)
#flat_qBOLD=layers.Activation('linear', dtype='float32')(flat_qBOLD)
#flat_QSM=layers.Activation('linear', dtype='float32')(flat_QSM)


grid_search_layer = layers.Lambda(QQfunc.grid_search_wrapper, name = 'grid_search')([flat_S0,flat_R2,flat_Y,flat_nu,flat_chinb,flat_qBOLD,flat_QSM])

#grid_search_layer = tf.debugging.check_numerics(grid_search_layer, message="check grid_search_layer")

conv_grid_search_1 = layers.Conv2D(filters = 8,
                                   kernel_size=(3,3),
                                   strides=(2,2),
                                   activation="tanh",
                                   #kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                   #bias_regularizer=tf.keras.regularizers.l1(0.01),
                                   name="conv_grid_1")(grid_search_layer)

#drop_grid_search_1 = layers.Dropout(0.1)(conv_grid_search_1)

conv_grid_search_2 = layers.Conv2D(filters = 16,
                                   kernel_size=(3,3),
                                   strides=(2,2),
                                   activation="tanh",
                                   #kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                   #bias_regularizer=tf.keras.regularizers.l1(0.01),
                                   name="conv_grid_2")(conv_grid_search_1)

#drop_grid_search_2 = layers.Dropout(0.1)(conv_grid_search_2)

#reshape_conv_grid_search_2 = layers.Reshape((-1,4,16),name='reshape_conv_grid_2')(conv_grid_search_2)


conv_grid_search_3 = layers.Conv2D(filters = 32,
                                   kernel_size=(4,4),
                                   strides=(1,1),
                                   activation="tanh",
                                   #kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                   #bias_regularizer=tf.keras.regularizers.l1(0.01),
                                   name="conv_grid_3")(conv_grid_search_2)

#drop_grid_search_3 = layers.Dropout(0.1)(conv_grid_search_3)


reshape_grid_search = layers.Reshape((30,30,32),name='collapse_parameter_space')(conv_grid_search_3)

#conv_S0_grid = layers.Conv2D(1,3,padding='same',activation="linear", name = 'S0_grid')(    reshape_grid_search)
#conv_R2_grid = layers.Conv2D(1,3,padding='same',activation="linear", name = 'R2_grid')(    reshape_grid_search)
conv_Y_grid = layers.Conv2D(1,3,padding='same',activation="linear", name = 'Y_grid')(     reshape_grid_search)
calc_nu_grid = layers.Lambda(QQfunc.f_nu_tensor_2, name="nu_calc")([conv_Y_grid,output_Params[4],model_params.input[1]])
#conv_nu_grid = layers.Conv2D(1,3,padding='same',activation="linear", name = 'nu_grid')(    reshape_grid_search)
#conv_chinb_grid = layers.Conv2D(1,3,padding='same',activation="linear", name = 'chi_nb_grid')(reshape_grid_search)



#model_grid_search = keras.Model(inputs=[model_params.input[0],model_params.input[1]],outputs=[conv_S0_grid,conv_R2_grid,conv_Y_grid,conv_nu_grid,conv_chinb_grid],name="grid_search_model")
model_grid_search = keras.Model(inputs=[model_params.input[0],model_params.input[1]],outputs=[conv_Y_grid,calc_nu_grid],name="grid_search_model")
#model_grid_search = keras.Model(inputs=[model_params.input[0],model_params.input[1]],outputs=conv_grid_search_2,name="grid_search_model")
model_grid_search.summary()
keras.utils.plot_model(model_grid_search, show_shapes=True)

#%%
def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes
get_model_memory_usage(50,model_grid_search)
#%%

opt = keras.optimizers.Adam(0.001, clipvalue=10.)
#loss=keras.losses.MeanAbsolutePercentageError()
#loss=keras.losses.MeanSquaredLogarithmicError()
loss=keras.losses.MeanAbsoluteError()
#loss=tf.keras.losses.Huber()
losses = {
    #"S0_grid":loss,
    #"R2_grid":loss,
    "Y_grid":loss,
    "nu_calc":loss,
    #"chi_nb_grid":loss,
}
lossWeights = {
    #"S0_grid":1.0,
    #"R2_grid":1.0,
    "Y_grid":1.0,
    "nu_calc":1.0,
    #"chi_nb_grid":1.0,
}

#model_params.trainable = False
model_grid_search.compile(
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
    tf.keras.callbacks.ModelCheckpoint(filepath='model_grid_search_3.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs/grid_search_3',profile_batch=[1,3])
]




history_grid_search = model_grid_search.fit([qBOLD_training,QSM_training], training_list[2:4] , batch_size=8, epochs=100, validation_split=0.1/0.9, callbacks=my_callbacks)

model_grid_search.save("models/"+version+ "Model_2D_params_grid_search_3.h5")
np.save('models/'+version+'history_params_2D_grid_search_3.npy',history_grid_search.history)

# %%

model_grid_search = keras.models.load_model("models/"+version+ "Model_2D_params_grid_search_3.h5")
model_grid_search.summary()
keras.utils.plot_model(model_grid_search, show_shapes=True)
# %%
p_grid = model_grid_search.predict([qBOLD_test,QSM_test])
p_grid[0].shape
len(p_grid)
p_combined = [p[0],p[1],p_grid[0],p_grid[1],p[4]]
#%%
Number=2
label_transformed=QQplt.translate_Params(test_list)
prediction_transformed_grid=QQplt.translate_Params(p_combined)

QQplt.check_Params_transformed(label_transformed,prediction_transformed_grid,Number,'CNN_Uniform_GESFIDE_16Echoes_grid_search_3')

QQplt.check_Params_transformed_hist(label_transformed,prediction_transformed_grid,'CNN_Uniform_GESFIDE_16Echoes_grid_search_3_evaluation')

QQplt.check_nu_calc(label_transformed,prediction_transformed_grid,QSM_test)
QQplt.check_nu_calc_QSM_noiseless(label_transformed,prediction_transformed_grid,test_list)



#%%

""" Second training step """

#qBOLD_layer = layers.Lambda(f_qBOLD_tensor, name = 'qBOLD')([dense_layer_3a,dense_layer_3b,dense_layer_3c,dense_layer_3d,dense_layer_3e])
qBOLD_layer = layers.Lambda(QQfunc.f_qBOLD_tensor, name = 'qBOLD')(model_params.output)
#QSM_layer = layers.Lambda(f_QSM_tensor, name = 'QSM')([dense_layer_3c,dense_layer_3d,dense_layer_3e])
QSM_layer = layers.Lambda(QQfunc.f_QSM_tensor, name = 'QSM')(model_params.output[2:5])
model_params.output[2:5]

model = keras.Model(inputs=[input_qBOLD,input_QSM],outputs=[qBOLD_layer,QSM_layer],name="Lambda_model")
model.summary()
keras.utils.plot_model(model, show_shapes=True)

# %% Train full model
opt = keras.optimizers.Adam(0.001, clipnorm=1.)
loss=keras.losses.MeanSquaredError()
losses = {
    "qBOLD":loss,
    "QSM":loss,
}
lossWeights = {
    "qBOLD":1.0,
    "QSM":10.0,
    }

model.compile(
    loss=losses,
    loss_weights=lossWeights,
    optimizer=opt,
    #metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
    #metrics=["accuracy"],
)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3),
    tf.keras.callbacks.ModelCheckpoint(filepath='Model_2D_fully_conv_direkt_Full_simple_tanh_sigmoid.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]
history = model.fit([qBOLD_training,QSM_training], [qBOLD_training,QSM_training] , batch_size=200, epochs=100, validation_split=0.1/0.9, callbacks=my_callbacks)

test_scores = model.evaluate([qBOLD_test,QSM_test],  [qBOLD_test,QSM_test], verbose=2)
#qBOLD_test.shape
#qBOLD_test2 = tf.expand_dims(qBOLD_test,axis=3)
#qBOLD_test2.shape
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
test_scores_params = model_params.evaluate([qBOLD_test,QSM_test], test_list, verbose=2)

#%%
#model_params.save("models/"+version+ "Model_2D_fully_conv_Params_after_qqbold_5_times_weights_10_times_QSM.h5")
#model.save("models/"+version+ "Model_2D_fully_conv_Full_qqbold_5_times_weights_10_times_QSM.h5")
#model_params.save("models/"+version+ "Model_2D_fully_conv_direkt_Params_simple_tanh_linear.h5")
#model.save("models/"+version+ "Model_2D_fully_conv_direkt_Full_simple_tanh_linear.h5")
np.save('models/'+version+'history_params_2D_direkt_Full_simple_tanh_sigmoid.npy',history.history)




# %%
print(history.history.keys())
QQplt.plot_loss(history,'')
QQplt.plot_loss(history,'qBOLD_')
QQplt.plot_loss(history,'QSM_')


#%%
model=tf.keras.models.load_model('Model_2D_fully_conv_direkt_Full_simple_tanh_linear.12-0.00.h5')
model.layers

#layer1=model.get_layer(index=1)
keras.utils.plot_model(model, show_shapes=True)


model_params_new= keras.models.Model(inputs=model.input,
                                    outputs=[model.get_layer('S0').output,
                                            model.get_layer('R2').output,
                                            model.get_layer('Y').output,
                                            model.get_layer('nu').output,
                                            model.get_layer('chi_nb').output])

keras.utils.plot_model(model_params_new, show_shapes=True)
#%%
p_after = model_params.predict([qBOLD_test,QSM_test])
#%%
Number = 2

check_Params(test_list,p_after,Number)

#%%
Number=2
label_transformed=QQplt.translate_Params(test_list)
prediction_transformed=QQplt.translate_Params(p_after)
QQplt.check_Params_transformed(label_transformed,prediction_transformed,Number)

QQplt.check_Params_transformed_hist(label_transformed,prediction_transformed)


#%%

p_full = model.predict([qBOLD_test,QSM_test])
QSM_test.shape
len(p_full)
p_full[1].shape
QQplt.check_QSM(QSM_test,p_full[1],Number)


#%%
qBOLD_test.shape
len(p_full)
p_full[0].shape
QQplt.check_qBOLD(qBOLD_test,p_full[0],Number)


#%% check qBOLD Verlauf

Number=2
QQplt.check_Pixel(qBOLD_test,p_full[0],QSM_test,p_full[1],Number)
