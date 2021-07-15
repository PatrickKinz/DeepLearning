# %% import modules
import tensorflow as tf
tf.config.list_physical_devices('GPU')
import numpy as np
import h5py
from numpy.random import rand, randn
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# %%

f = h5py.File("Exponential2D_bigger.hdf5", "r")
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

conv_layer_2 = keras.layers.Conv2D(2*n,
                  (3,3),
                  strides=(1,1),
                  padding='same',
                  dilation_rate=1,
                  activation='relu',
                  name='conv_layer_2')(conv_layer_1)

dense_layer_3a = layers.Dense(1, name = 'Dense_3a_S0')(conv_layer_2) # 3 outputs for S0, T2 and T2S
dense_layer_3b = layers.Dense(1, name = 'Dense_3b_T2')(conv_layer_2) # 3 outputs for S0, T2 and T2S
dense_layer_3c = layers.Dense(1, name = 'Dense_3c_T2S')(conv_layer_2) # 3 outputs for S0, T2 and T2S

#before_lambda_model = keras.Model(input_layer, dense_layer_3, name="before_lambda_model")

Params_Layer = layers.Concatenate(name = 'Output_Params')([dense_layer_3a,dense_layer_3b,dense_layer_3c])
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


history = model_params.fit(input_noise_train, target_train, batch_size=50, epochs=50, validation_split=0.2, callbacks=my_callbacks)
test_scores = model_params.evaluate(input_noise_test, target_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

# %%
p = model_params.predict(input_noise_test)

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
ax[3].title.set_text('S0')
ax[4].imshow(p[Number,:,:,1], cmap='Greys')
ax[4].title.set_text('T2')
ax[5].imshow(p[Number,:,:,2], cmap='Greys')
ax[5].title.set_text('T2S')
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=3)
ax = axes.ravel()
ax[0].hist(target_test[Number,:,:,0].ravel())
ax[0].title.set_text('S0')
ax[1].hist(target_test[Number,:,:,1].ravel())
ax[1].title.set_text('T2')
ax[2].hist(target_test[Number,:,:,2].ravel())
ax[2].title.set_text('T2S')
ax[3].hist(p[Number,:,:,0].ravel())
ax[3].title.set_text('S0')
ax[4].hist(p[Number,:,:,1].ravel())
ax[4].title.set_text('T2')
ax[5].hist(p[Number,:,:,2].ravel())
ax[5].title.set_text('T2S')
plt.show()
# ax[3].hist(target_test[Number,:,:,0].ravel())
# ax[4].hist(target_test[Number,:,:,1].ravel())
# ax[5].hist(target_test[Number,:,:,2].ravel())
