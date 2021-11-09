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
import h5py
#from QSM_qBOLD_2D_load_and_prepare_data import load_and_prepare_data

#tf.keras.mixed_precision.set_global_policy("mixed_float16") #accelerates training, expecially with tensor cores on RTX cards
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
#%%

conv_QSM_denoise = keras.layers.Conv2D(n,
                  kernel_size=5,
                  strides=(1),
                  padding='same',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_QSM_denoise')(input_QSM)

conv_QSM_denoise_out = keras.layers.Conv2D(1,
                  kernel_size=3,
                  strides=(1),
                  padding='same',
                  dilation_rate=1,
                  activation='tanh',
                  name='conv_QSM_denoise_out')(conv_QSM_denoise)



model_QSM_denoise = keras.Model(inputs=input_QSM, outputs = conv_QSM_denoise_out, name="QSM_denoise_model")
model_QSM_denoise.summary()
keras.utils.plot_model(model_QSM_denoise, show_shapes=True)


# %%
concat_QQ_1 = layers.Concatenate(name = 'concat_QQ_1')([model_qBOLD.output,model_QSM.output])

conv_QQ_1 = layers.Conv2D(2*n,3,padding='same',activation="tanh",name = 'conv_QQ_1')(concat_QQ_1)


conv_S0 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'S0')(    conv_QQ_1)
conv_R2 = layers.Conv2D(1,3,padding='same',activation="linear", name = 'R2')(    conv_QQ_1)
conv_chinb = layers.Conv2D(1,3,padding='same',activation="linear", name = 'chi_nb')(conv_QQ_1)

model_params = keras.Model(inputs=[input_qBOLD,input_QSM],outputs=[conv_S0,conv_R2,conv_chinb],name="Params_model")
model_params.summary()
keras.utils.plot_model(model_params, show_shapes=True)

#%%
def f_nu_tensor(tensor):
    c = tensor[0]
    e = tensor[1]
    QSM=tensor[2]

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
    chi_nb = ( 0.1-(-0.1) ) * e - 0.1

    nenner = (chi_ba-chi_nb)/alpha + psi_Hb*delta_chi_Hb * ((1-(1-alpha)*SaO2)/alpha - Y)
    nu = (QSM - chi_nb) / nenner
    d = (nu-0.001)/(0.1-0.001) # <=>  nu = (0.1 - 0.001) * d + 0.001
    return d
#%%
def f_remove_S0_R2(tensor):
    t=tf.constant([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], dtype=tf.float32)*3/1000
    qBOLD=tensor[0]
    S0 = tensor[1]
    R2=tensor[2]

    return  tf.math.divide_no_nan(qBOLD,S0)*tf.math.exp(R2*t)


# %%

remove_S0_R2_layer= layers.Lambda(f_remove_S0_R2,name='remove_S0_R2')([input_qBOLD,model_params.output[0],model_params.output[1]])

concat_QQ_2 = layers.Concatenate(name = 'concat_QQ_2')([model_params.output[0],model_params.output[1],model_params.output[2],remove_S0_R2_layer])
conv_QQ_2 = layers.Conv2D(4*n,3,padding='same',activation="tanh",name = 'conv_QQ_2')(concat_QQ_2)


conv_Y = layers.Conv2D(1,3,padding='same',activation="linear", name = 'Y')(     conv_QQ_2)
conv_nu = layers.Conv2D(1,3,padding='same',activation="linear", name = 'nu')(    conv_QQ_2)


model_params_remove_R2_S0 = keras.Model(inputs=[input_qBOLD,input_QSM],outputs=[conv_S0,conv_R2,conv_Y,conv_nu,conv_chinb],name="Params_model_remove_R2_S0")
model_params_remove_R2_S0.summary()
keras.utils.plot_model(model_params_remove_R2_S0, show_shapes=True)



#%%
limit_Y =layers.ReLU(max_value=1,name='Y_limited')(conv_Y)
limit_chinb =layers.ReLU(max_value=1,name='chi_nb_limited')(conv_chinb)
nu_func_layer = layers.Lambda(f_nu_tensor, name = 'nu_func')([limit_Y,limit_chinb,model_QSM_denoise.output])
limit_nu = layers.ReLU(max_value=1,name='nu_limited')(nu_func_layer)
concat_QQ_3 = layers.Concatenate(name = 'concat_QQ_3')([limit_Y,limit_nu,conv_nu,limit_chinb,remove_S0_R2_layer,model_QSM_denoise.output])
conv_QQ_3 = layers.Conv2D(4*n,3,padding='same',activation="tanh",name = 'conv_QQ_3')(concat_QQ_3)

conv_Y_final = layers.Conv2D(1,3,padding='same',activation="linear", name = 'Y_final')(     conv_QQ_3)
conv_nu_final = layers.Conv2D(1,3,padding='same',activation="linear", name = 'nu_final')(    conv_QQ_3)


model_params_explicit_QSM_reduced_S0_R2 = keras.Model(inputs=[input_qBOLD,input_QSM],outputs=[conv_S0,conv_R2,conv_Y_final,conv_nu_final,limit_chinb],name="Params_model_explicit_QSM_reduced_S0_R2")
model_params_explicit_QSM_reduced_S0_R2.summary()
keras.utils.plot_model(model_params_explicit_QSM_reduced_S0_R2, show_shapes=True)



# %% Train Params model

opt = keras.optimizers.Adam(0.001, clipnorm=1.)
#loss=keras.losses.MeanAbsolutePercentageError()
#loss=keras.losses.MeanSquaredLogarithmicError()
loss=keras.losses.MeanAbsoluteError()
#loss=tf.keras.losses.Huber()
losses = {
    "S0":loss,
    "R2":loss,
    "chi_nb":loss,
}
lossWeights = {
    "S0":1.0,
    "R2":1.0,
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
opt = keras.optimizers.Adam(0.001, clipnorm=1.)
model_QSM_denoise.compile(
    loss=keras.losses.MeanAbsoluteError(),
    optimizer=opt,
    #metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
    #metrics=[tf.keras.metrics.MeanSquaredError()],
    #metrics=["accuracy"],
)

#%%
opt = keras.optimizers.Adam(0.001, clipnorm=1.)
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
model_params_remove_R2_S0.compile(
    loss=losses,
    loss_weights=lossWeights,
    optimizer=opt,
    #metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
    #metrics=[tf.keras.metrics.MeanSquaredError()],
    #metrics=["accuracy"],
)

#%%
opt = keras.optimizers.Adam(0.001, clipnorm=1.)
losses = {
    "S0":loss,
    "R2":loss,
    "Y_final":loss,
    "nu_final":loss,
    "chi_nb_limited":loss,
}
lossWeights = {
    "S0":1.0,
    "R2":1.0,
    "Y_final":1.0,
    "nu_final":1.0,
    "chi_nb_limited":1.0,
}
model_params_explicit_QSM_reduced_S0_R2.compile(
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

history_QSM_denoise=model_QSM_denoise.fit(QSM_training,QSM_training, batch_size=100, epochs=5, validation_split=0.1/0.9, callbacks=my_callbacks)
#%%
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]

training_list = [S0_train,R2_train,chi_nb_train]

history_params = model_params.fit([qBOLD_training,QSM_training], training_list , batch_size=100, epochs=10, validation_split=0.1/0.9, callbacks=my_callbacks)
#%%
my_callbacks = [
    #tf.keras.callbacks.EarlyStopping(patience=3),
    #tf.keras.callbacks.ModelCheckpoint(filepath='models\model_params_remove_R2_S0.{epoch:02d}-{val_loss:.4f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]

training_list = [S0_train,R2_train,Y_train,nu_train,chi_nb_train]

history_params_remove_R2_S0 = model_params_remove_R2_S0.fit([qBOLD_training,QSM_training], training_list , batch_size=100, epochs=10, validation_split=0.1/0.9, callbacks=my_callbacks)

#%%
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3),
    #tf.keras.callbacks.ModelCheckpoint(filepath='models\model_params_remove_R2_S0.{epoch:02d}-{val_loss:.4f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]

training_list = [S0_train,R2_train,Y_train,nu_train,chi_nb_train]

history_params_explicit_QSM_remove_R2_S0 = model_params_explicit_QSM_reduced_S0_R2.fit([qBOLD_training,QSM_training], training_list , batch_size=100, epochs=100, validation_split=0.1/0.9, callbacks=my_callbacks)



#history_params = model_params.fit(training_Params_data, epochs=100,validation_data=val_Params_data, callbacks=my_callbacks)
#%%
#model_params.save("models/"+version+ "Model_2D_fully_conv_Params_before_qqbold_simple_tanh_linear.h5")
#np.save('models/'+version+'history_params_2D_fully_conv_Params_before_qqbold_simple_tanh_linear.npy',history_params.history)
#model_params = keras.models.load_model("models/"+version+ "Model_2D_fully_conv_Params_before_qqbold_simple_tanh_linear.h5")

#%%

test_list = [S0_test,R2_test,Y_test,nu_test,chi_nb_test]


test_scores = model_params_explicit_QSM_reduced_S0_R2.evaluate([qBOLD_test,QSM_test], test_list, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
#%%
print(history_params_explicit_QSM_remove_R2_S0.history.keys())
def plot_loss(history, keyword):
    plt.figure()
    plt.plot(history.history[keyword + 'loss'])
    plt.plot(history.history['val_'+keyword+'loss'])
    plt.yscale('log')
    plt.title('model ' +keyword+ 'loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

plot_loss(history_params_explicit_QSM_remove_R2_S0,'')
plot_loss(history_params_explicit_QSM_remove_R2_S0,'S0_')
plot_loss(history_params_explicit_QSM_remove_R2_S0,'R2_')
plot_loss(history_params_explicit_QSM_remove_R2_S0,'Y_final_')
plot_loss(history_params_explicit_QSM_remove_R2_S0,'nu_final_')
plot_loss(history_params_explicit_QSM_remove_R2_S0,'chi_nb_limited_')


# %%
#model_params_explicit_QSM = keras.models.load_model("model_params_explicit_QSM.07-0.4561.h5")
#model_params.summary()
p = model_params_explicit_QSM_reduced_S0_R2.predict([qBOLD_test,QSM_test])
p[0].shape

#%%
Number=2
label_transformed=QQplt.translate_Params(test_list)
prediction_transformed=QQplt.translate_Params(p)
QQplt.check_Params_transformed(label_transformed,prediction_transformed,Number)

QQplt.check_Params_transformed_hist(label_transformed,prediction_transformed)

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
    tf.keras.callbacks.ModelCheckpoint(filepath='Model_2D_fully_conv_direkt_Full_simple_tanh_linear.{epoch:02d}-{val_loss:.2f}.h5'),
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
np.save('models/'+version+'history_params_2D_direkt_Full_simple_tanh_linear.npy',history.history)




# %%
print(history.history.keys())
plot_loss(history,'')
plot_loss(history,'qBOLD_')
plot_loss(history,'QSM_')


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
p_after = model_params_new.predict([qBOLD_test,QSM_test])
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
QQplt.check_QSM(QSM_test,p_full[1],Number)


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
QQplt.check_qBOLD(qBOLD_test,p_full[0],Number)


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
QQplt.check_Pixel(qBOLD_test,p_full[0],QSM_test,p_full[1],Number)
