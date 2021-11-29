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

Dataset_train=np.load("../Brain_Phantom/Patches_no_air_big/15GB_1Pnoise_train_val.npz")
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
Dataset_test=np.load("../Brain_Phantom/Patches_no_air_big/15GB_1Pnoise_test.npz")
S0_test=Dataset_test['S0']
S0_test.shape
R2_test=Dataset_test['R2']
Y_test=Dataset_test['Y']
nu_test=Dataset_test['nu']
chi_nb_test=Dataset_test['chi_nb']
qBOLD_test=Dataset_test['qBOLD']
QSM_test=Dataset_test['QSM']

test_list = [S0_test,R2_test,Y_test,nu_test,chi_nb_test]


version = "no_air_1Pnoise_15GB/"

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





model_qBOLD = keras.Model(inputs=input_qBOLD, outputs = conv_qBOLD_1, name="qBOLD model")
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


model_QSM = keras.Model(inputs=input_QSM, outputs = conv_QSM_1, name="QSM model")
model_QSM.summary()
keras.utils.plot_model(model_QSM, show_shapes=True)

concat_QQ_1 = layers.Concatenate(name = 'concat_QQ_1')([model_qBOLD.output,model_QSM.output])
conv_QQ_1 = layers.Conv2D(2*n,3,padding='same',activation="tanh",name = 'conv_QQ_1')(concat_QQ_1)
#conv_QQ_1 = layers.Conv2D(2*n,3,padding='same',activation="tanh",name = 'conv_QQ_1')(model_qBOLD.output)



conv_S0 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'S0')(    conv_QQ_1)
conv_R2 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'R2')(    conv_QQ_1)
conv_Y = layers.Conv2D(1,3,padding='same',activation="linear", name = 'Y')(     conv_QQ_1)
conv_nu = layers.Conv2D(1,3,padding='same',activation="linear", name = 'nu')(    conv_QQ_1)
conv_chinb = layers.Conv2D(1,3,padding='same',activation="linear", name = 'chi_nb')(conv_QQ_1)


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
#model_params.save("models/"+version+ "Model_2D_fully_conv_Params_before_qqbold_simple_tanh_linear.h5")
#np.save('models/'+version+'history_params_2D_fully_conv_Params_before_qqbold_simple_tanh_linear.npy',history_params.history)
model_params = keras.models.load_model("models/"+version+ "Model_2D_fully_conv_Params_before_qqbold_simple_tanh_linear.h5")
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
Number=2
label_transformed=QQplt.translate_Params(test_list)
prediction_transformed=QQplt.translate_Params(p)
QQplt.check_Params_transformed(label_transformed,prediction_transformed,Number,'CNN_Uniform_GESFIDE_16Echoes_Params')

QQplt.check_Params_transformed_hist(label_transformed,prediction_transformed,'CNN_Uniform_GESFIDE_16Echoes_evaluation')

QQplt.check_nu_calc(label_transformed,prediction_transformed,QSM_test)
QQplt.check_nu_calc_QSM_noiseless(label_transformed,prediction_transformed,test_list)


nu_calc = QQfunc.f_nu(prediction_transformed[2],prediction_transformed[4],QSM_test)
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
