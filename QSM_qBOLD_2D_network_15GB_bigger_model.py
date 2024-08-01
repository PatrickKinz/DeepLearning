# %% import modules
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
from numpy.random import rand, randn,shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm  #for progress bar
import QSM_qBOLD_2D_plotting_functions as QQplt
import QSM_and_qBOLD_functions as QQfunc
import h5py
#from QSM_qBOLD_2D_load_and_prepare_data import load_and_prepare_data

#tf.keras.mixed_precision.set_global_policy("mixed_float16") #accelerates training, expecially with tensor cores on RTX cards
#from My_Custom_Generator import My_Params_Generator,My_Signal_Generator
#%%
#data_dir = "../Brain_Phantom/Patches/"
#Params_training,Params_test,qBOLD_training,qBOLD_test,QSM_training,QSM_test = load_and_prepare_data(data_dir)

#np.savez("../Brain_Phantom/Patches/NumpyArchiv",Params_training=Params_training,Params_test=Params_test,qBOLD_training=qBOLD_training,qBOLD_test=qBOLD_test,QSM_training=QSM_training,QSM_test=QSM_test)

Dataset_train=np.load("../Brain_Phantom/Patches_no_air_big/15GB_0noise_train_val.npz")
S0_train=Dataset_train['S0']
S0_train.shape
R2_train=Dataset_train['R2']
Y_train=Dataset_train['Y']
nu_train=Dataset_train['nu']
chi_nb_train=Dataset_train['chi_nb']
qBOLD_training=Dataset_train['qBOLD']
QSM_training=Dataset_train['QSM']


training_list = [S0_train,R2_train,Y_train,nu_train,chi_nb_train]

#%%
Dataset_test=np.load("../Brain_Phantom/Patches_no_air_big/15GB_0noise_test.npz")
S0_test=Dataset_test['S0']
S0_test.shape
R2_test=Dataset_test['R2']
Y_test=Dataset_test['Y']
nu_test=Dataset_test['nu']
chi_nb_test=Dataset_test['chi_nb']
qBOLD_test=Dataset_test['qBOLD']
QSM_test=Dataset_test['QSM']

test_list = [S0_test,R2_test,Y_test,nu_test,chi_nb_test]


version = "no_air_0noise_15GB/"

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
                  activation='tanh',
                  name='conv_qBOLD_1')(input_qBOLD)

conv_qBOLD_2 = keras.layers.Conv2D(2*n,
                  kernel_size = 3,
                  strides=1,
                  padding='same',
                  dilation_rate=1,
                  activation='tanh',
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
                  activation='tanh',
                  name='conv_QSM_1')(input_QSM)


model_QSM = keras.Model(inputs=input_QSM, outputs = conv_QSM_1, name="QSM model")
model_QSM.summary()
keras.utils.plot_model(model_QSM, show_shapes=True)
# %%
concat_QQ_1 = layers.Concatenate(name = 'concat_QQ_1')([model_qBOLD.output,model_QSM.output])
conv_QQ_1 = layers.Conv2D(2*n,3,padding='same',activation="tanh",name = 'conv_QQ_1')(concat_QQ_1)
#conv_QQ_1 = layers.Conv2D(2*n,3,padding='same',activation="tanh",name = 'conv_QQ_1')(model_qBOLD.output)

conv_QQ_2 = layers.Conv2D(4*n,3,padding='same',activation="tanh",name = 'conv_QQ_2')(conv_QQ_1)


conv_S0 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'S0')(    conv_QQ_2)
conv_R2 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'R2')(    conv_QQ_2)
conv_Y = layers.Conv2D(1,3,padding='same',activation="linear", name = 'Y')(     conv_QQ_2)
conv_nu = layers.Conv2D(1,3,padding='same',activation="linear", name = 'nu')(    conv_QQ_2)
conv_chinb = layers.Conv2D(1,3,padding='same',activation="linear", name = 'chi_nb')(conv_QQ_2)


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




history_params = model_params.fit([qBOLD_training,QSM_training], training_list , batch_size=100, epochs=100, validation_split=0.1/0.9, callbacks=my_callbacks)
#history_params = model_params.fit(training_Params_data, epochs=100,validation_data=val_Params_data, callbacks=my_callbacks)
#%%
model_params.save("models/"+version+ "Model_2D_fully_conv_Params_before_qqbold_bigger_tanh_linear.h5")
np.save('models/'+version+'history_params_2D_fully_conv_Params_before_qqbold_bigger_tanh_linear.npy',history_params.history)
#model_params = keras.models.load_model("models/"+version+ "Model_2D_fully_conv_Params_before_qqbold_simple_tanh_linear.h5")
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
QQplt.check_Params_transformed(label_transformed,prediction_transformed,Number,'CNN_Uniform_GESFIDE_16Echoes_0noise_bigger_Params')

QQplt.check_Params_transformed_hist(label_transformed,prediction_transformed,'CNN_Uniform_GESFIDE_16Echoes_0noise_bigger_evaluation')

#%%

nu_calc = QQfunc.f_nu(prediction_transformed[2],prediction_transformed[4],QSM_test)

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(5,5))
counts, xedges, yedges, im = axes.hist2d(x=label_transformed[3][:,:,:].ravel()*100,y=np.squeeze(nu_calc[:,:,:,:]).ravel()*100,bins=30,range=((0,10),(-5,15)),cmap='inferno')
axes.title.set_text('$v$ [%]')
axes.set_xlabel('truth')
axes.set_ylabel('calculation')
cbar=fig.colorbar(im,ax=axes)
cbar.formatter.set_powerlimits((0, 0))
axes.plot(np.linspace(0,10,10),np.linspace(0,10,10))
plt.show()
#%%
Number=2
QSM_test.shape
plt.figure()
plt.imshow(QSM_test[Number,:,:,0], cmap='gray')




# %%
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
