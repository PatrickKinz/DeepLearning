# %% import modules
import tensorflow as tf
tf.config.list_physical_devices('GPU')
import numpy as np
import h5py
from numpy.random import rand, randn
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sys import getsizeof

# %%

f = h5py.File("data/Exponential2D_bigger.hdf5", "r")
list(f.keys())
dset_input_train = f["input_train"]
dset_input_noise_train = f["input_noise_train"]
dset_target_train = f["target_train"]
dset_input_test = f["input_test"]
dset_input_noise_test = f["input_noise_test"]
dset_target_test = f["target_test"]

input_train = np.array(dset_input_train)
dset_input_train.shape
dset_input_train.dtype
input_noise_train = np.array(dset_input_noise_train)
target_train = np.array(dset_target_train)
input_test = np.array(dset_input_test)
input_noise_test = np.array(dset_input_noise_test)
target_test = np.array(dset_target_test)
# %%
f.close()


input_train.shape
input_train.dtype

plt.plot(input_train[0,1,1,:],'o-')
plt.plot(input_noise_train[0,1,1,:],'o')

# %% norm

def norm_signal_array(input,target):
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            for k in range(input.shape[2]):
                scale = input[i,j,k,0]
                input[i,j,k,:] =  input[i,j,k,:]/scale
                target[i,j,k,0] =  target[i,j,k,0]/scale
    return input,target
input_noise_norm_train, target_train = norm_signal_array(input_noise_train, target_train)
plt.plot(input_noise_norm_train[0,1,1,:],'o')
input_noise_norm_test, target_test = norm_signal_array(input_noise_test, target_test)

"""
input_noise_norm_train = np.log(input_noise_train)
input_noise_norm_test = np.log(input_noise_test)
input_norm_test = np.log(input_test)

target_train[:,:,:,0] = np.log(target_train[:,:,:,0])
target_test[:,:,:,0] = np.log(target_test[:,:,:,0])
"""
"""
input_noise_norm_train = input_noise_train
input_noise_norm_test = input_noise_test
input_norm_test = input_test
"""
# %% augment data
def augment_data(m):
    m = np.concatenate((m,np.flip(m, axis=1)), axis=0)
    m = np.concatenate((m,np.rot90(m, k=1, axes=(2,1))), axis=0)
    m = np.concatenate((m,np.flip(m, axis=(1,2))), axis=0)
    return m


#input_train = augment_data(input_train)
input_noise_norm_train = augment_data(input_noise_norm_train)
getsizeof(input_noise_norm_train)/(1000*1000*1000) # GB
target_train = augment_data(target_train)

input_test = augment_data(input_test)
input_noise_norm_test = augment_data(input_noise_norm_test)
target_test = augment_data(target_test)


# %% Look at rotation
fig, axes = plt.subplots(nrows=2, ncols=2)
ax = axes.ravel()
ax[0].imshow(input_noise_norm_test[0,:,:,1], cmap='Greys')
ax[1].imshow(input_noise_norm_test[100,:,:,1], cmap='Greys')
ax[2].imshow(input_noise_norm_test[200,:,:,1], cmap='Greys')
ax[3].imshow(input_noise_norm_test[300,:,:,1], cmap='Greys')
fig, axes = plt.subplots(nrows=2, ncols=2)
ax = axes.ravel()
ax[0].imshow(input_noise_norm_test[400,:,:,1], cmap='Greys')
ax[1].imshow(input_noise_norm_test[500,:,:,1], cmap='Greys')
ax[2].imshow(input_noise_norm_test[600,:,:,1], cmap='Greys')
ax[3].imshow(input_noise_norm_test[700,:,:,1], cmap='Greys')







# %% Network

input_layer = keras.Input(shape=(128,128,16), name = 'Input_layer')
input_layer.shape
input_layer.dtype

n=16

conv_layer_1 = keras.layers.Conv2D(n,
                  (3,3),
                  strides=(1,1),
                  padding='same',
                  dilation_rate=1,
                  activation='relu',
                  name='conv_layer_1')(input_layer)

conv_layer_2 = keras.layers.Conv2D(n,
                  (3,3),
                  strides=(1,1),
                  padding='same',
                  dilation_rate=1,
                  activation='relu',
                  name='conv_layer_2')(conv_layer_1)

concatenate_layer = layers.Concatenate(name = 'Concat_Layer')([input_layer,conv_layer_2])
dense_layer_1 = layers.Dense(16,name = 'Dense_1')(concatenate_layer)
dense_layer_2 = layers.Dense(8,name = 'Dense_2')(dense_layer_1)

dense_layer_3a = layers.Dense(1, name = 'Dense_3a_S0')(dense_layer_2) # 3 outputs for S0, T2 and T2S
dense_layer_3b = layers.Dense(1, name = 'Dense_3b_T2')(dense_layer_2) # 3 outputs for S0, T2 and T2S
dense_layer_3c = layers.Dense(1, name = 'Dense_3c_T2S')(dense_layer_2) # 3 outputs for S0, T2 and T2S

#before_lambda_model = keras.Model(input_layer, dense_layer_3, name="before_lambda_model")

Params_Layer = layers.Concatenate(name = 'Output_Params')([dense_layer_3a,dense_layer_3b,dense_layer_3c])
#Params_Layer = layers.Concatenate(name = 'Output_Params')([dense_layer_3b,dense_layer_3c])
model_params = keras.Model(inputs=input_layer,outputs=Params_Layer,name="Params_model")
model_params.summary()
keras.utils.plot_model(model_params, show_shapes=True)


# %% Train Params model
model_params.compile(
    loss=keras.losses.MeanAbsolutePercentageError(),
    optimizer='adam',
    metrics=["accuracy"],
)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]


history = model_params.fit(input_noise_norm_train, target_train[:,:,:,:], batch_size=10, epochs=50, validation_split=0.2, callbacks=my_callbacks)
test_scores = model_params.evaluate(input_noise_norm_test, target_test[:,:,:,:], verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])


model_params.save("Model_2D_Params_before.h5")

# %%
model_params = keras.models.load_model("Model_2D_Params_before.h5")
model_params.summary()

# %%
p = model_params.predict(input_noise_norm_test)

Number = 1

fig, axes = plt.subplots(nrows=2, ncols=3)
ax = axes.ravel()
ax[0].imshow(target_test[Number,:,:,0], cmap='Greys')
ax[0].title.set_text('S0')
ax[1].imshow(target_test[Number,:,:,1], cmap='Greys')
ax[1].title.set_text('T2')
ax[2].imshow(target_test[Number,:,:,2], cmap='Greys')
ax[2].title.set_text('T2S')
ax[3].imshow(p[Number,:,:,0], cmap='Greys')
#ax[3].title.set_text('S0')
ax[4].imshow(p[Number,:,:,1], cmap='Greys')
#ax[4].title.set_text('T2')
ax[5].imshow(p[Number,:,:,2], cmap='Greys')
#ax[5].title.set_text('T2S')
plt.show()
# %%
fig, axes = plt.subplots(nrows=1, ncols=3)
ax = axes.ravel()
ax[0].plot(target_test[Number,64,:,0])
ax[0].plot(p[Number,64,:,0])
ax[0].title.set_text('S0')
ax[1].plot(target_test[Number,64,:,1])
ax[1].plot(p[Number,64,:,1])
ax[1].title.set_text('T2')
ax[2].plot(target_test[Number,64,:,2])
ax[2].plot(p[Number,64,:,2])
ax[2].title.set_text('T2S')
plt.show()

# %%

fig, axes = plt.subplots(nrows=2, ncols=3)
ax = axes.ravel()
ax[0].hist(target_test[Number,:,:,0].ravel())
ax[0].title.set_text('S0')
ax[1].hist(target_test[Number,:,:,1].ravel())
ax[1].title.set_text('T2')
ax[2].hist(target_test[Number,:,:,2].ravel())
ax[2].title.set_text('T2S')
ax[3].hist(p[Number,:,:,0].ravel())
ax[4].hist(p[Number,:,:,1].ravel())
ax[5].hist(p[Number,:,:,2].ravel())
plt.show()
# ax[3].hist(target_test[Number,:,:,0].ravel())
# ax[4].hist(target_test[Number,:,:,1].ravel())
# ax[5].hist(target_test[Number,:,:,2].ravel())


# %%


def simulateSignal_for_FID(tensor):
    t=tf.constant([3,6,9,12,15,18], dtype=tf.float32)
    t.shape
    S0 = tensor[0]  #ln(S0)
    T2S = tensor[1]
    #output = S0 - tf.math.divide_no_nan(t,T2S)
    output = S0 * tf.math.exp(-tf.math.divide_no_nan(t,T2S))
    return output
"""
This kind of test does not work because of Dimensions, but it works somehow in the network
Incompatible shapes: [6] vs. [128,128] [Op:DivNoNan]

test_S0 = tf.convert_to_tensor(target_test[2,:,:,0])
test_S0.shape
test_T2 = tf.convert_to_tensor(target_test[2,:,:,1])
test_T2S = tf.convert_to_tensor(target_test[2,:,:,2])

out = simulateSignal_for_FID([test_S0,test_T2,test_T2S])
"""

def simulateSignal_for_Echo_Peak_rise(tensor):
    """x[0] = S0, x[1] = T2, x[2] = T2S   """
    t=tf.constant([21,24,27,30,33,36,39], dtype=tf.float32)
    S0 = tensor[0]
    T2 = tensor[1]
    T2S = tensor[2]
    #output = S0 - (40.0-t)*(tf.math.divide_no_nan(1.0,T2S) - tf.math.divide_no_nan(1.0,T2)) - tf.math.divide_no_nan(t,T2)
    output = S0 * tf.math.exp(- (40.0-t)*(tf.math.divide_no_nan(1.0,T2S) - tf.math.divide_no_nan(1.0,T2)) - tf.math.divide_no_nan(t,T2) )
    return output

def simulateSignal_for_Echo_Peak_fall(tensor):
    """x[0] = S0, x[1] = T2, x[2] = T2S   """
    t=tf.constant([42,45,48], dtype=tf.float32)
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
