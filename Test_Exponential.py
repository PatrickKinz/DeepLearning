# %% import modules
import tensorflow as tf
tf.config.list_physical_devices('GPU')
import numpy as np
from numpy.random import rand, randn
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import time
import sys
sys.path.append('C:/Users/pk24/Documents/Programming/Libraries/tf-levenberg-marquardt')
import levenberg_marquardt as lm

# %% Create Data
def simulateSignal(S0,T2,T2S,t):
    output = np.zeros((len(S0),len(t)))
    for i in range(6):
        output[:,i] = S0 * np.exp(-t[i]/T2S)
    for i in range(6,len(t)):
        output[:,i] = S0 * np.exp(-abs((40-t[i]))*(1/T2S - 1/T2) - t[i]/(T2) )
    return output


# create fake signal (16 points) 6 points decay, then refocus pulse, then 10 more points

def createData(N, t):
    S0 = rand(N) * 1000 + 1000
    #print(S0)
    #S0.shape
    T2 = randn(N)*5+60
    #print(T2)
    T2S = randn(N)*3+15
    #print(T2S)
    y=np.stack((S0,T2,T2S), axis = -1)
    #y.shape
    signal = simulateSignal(S0,T2,T2S,t)
    #
    signal = np.cast['float32'](signal)
    y = np.cast['float32'](y)
    return signal, y



N_train = 100000
N_test = 1000
t=np.array([3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48])

signal_train, y_train = createData(N_train,t)
signal_test, y_test = createData(N_test,t)
print(signal_train.shape)
print(y_train.shape)

plt.plot(t,np.transpose(signal_train[:5,:]), 'o-')

# %% Add noise to data

def addNoise(input,Spread,Offset):
    output = np.multiply(input, 1 + Spread*randn(input.size).reshape(input.shape)) + Offset*rand(input.size).reshape(input.shape)
    output = np.cast['float32'](output)
    return output

signal_train_noise = addNoise(signal_train, 0.02, 10)
print(signal_train_noise.shape)
signal_test_noise = addNoise(signal_test, 0.02, 10)

plt.plot(t,np.transpose(signal_train_noise[:5,:]), 'o-')

signal_test_noise.dtype
np.amin(signal_test_noise)
# %% Network

input_layer = keras.Input(shape=(16,), name = 'Input_layer')
input_layer.shape
input_layer.dtype

dense_layer_1 = layers.Dense(8, activation="relu", name = 'Dense_1')(input_layer)
dense_layer_1.shape
dense_layer_2 =layers.Dense(8,activation="relu", name = 'Dense_2')(dense_layer_1)
dense_layer_3a = layers.Dense(1, name = 'Dense_3a_S0')(dense_layer_2) # 3 outputs for S0, T2 and T2S
dense_layer_3b = layers.Dense(1, name = 'Dense_3b_T2')(dense_layer_2) # 3 outputs for S0, T2 and T2S
dense_layer_3c = layers.Dense(1, name = 'Dense_3c_T2S')(dense_layer_2) # 3 outputs for S0, T2 and T2S

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

model_params_wrapper = lm.ModelWrapper(
    tf.keras.models.clone_model(model_params))

model_params_wrapper.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
    loss=lm.ReducedOutputsMeanSquaredError())

# %%
print("Train using Adam")
t1_start = time.perf_counter()
history = model_params.fit(signal_train_noise, y_train, batch_size=50, epochs=10, validation_split=0.2)
t1_stop = time.perf_counter()
print("Elapsed time: ", t1_stop - t1_start)
test_scores = model_params.evaluate(signal_test_noise, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

# %%
print("Train using Levenberg-Marquardt")
#train_dataset = tf.data.Dataset.from_tensor_slices((signal_train_noise, signal_train_noise))
t2_start = time.perf_counter()
model_params_wrapper.fit(signal_train_noise, y_train, batch_size=500, epochs=100)
t2_stop = time.perf_counter()
print("Elapsed time: ", t2_stop - t2_start)




# %%
Number = 5

p = model_params.predict(signal_test)
print(y_test[Number,:])
print(p[Number,:])

model_params.save("Model_Params_before.h5")





# %%


def simulateSignal_for_FID(tensor):
    t=tf.constant([3,6,9,12,15,18], dtype=tf.float32)
    t.shape
    S0 = tensor[0]
    T2S = tensor[1]
    output = S0 * tf.math.exp(-tf.math.divide_no_nan(t,T2S))
    return output


def simulateSignal_for_Echo_Peak_rise(tensor):
    """x[0] = S0, x[1] = T2, x[2] = T2S   """
    t=tf.constant([21,24,27,30,33,36,39], dtype=tf.float32)
    S0 = tensor[0]
    T2 = tensor[1]
    T2S = tensor[2]
    output = S0 * tf.math.exp(- (40.0-t)*(tf.math.divide_no_nan(1.0,T2S) - tf.math.divide_no_nan(1.0,T2)) - tf.math.divide_no_nan(t,T2) )
    return output

def simulateSignal_for_Echo_Peak_fall(tensor):
    """x[0] = S0, x[1] = T2, x[2] = T2S   """
    t=tf.constant([42,45,48], dtype=tf.float32)
    S0 = tensor[0]
    T2 = tensor[1]
    T2S = tensor[2]
    output = S0 * tf.math.exp(- (t-40.0)*(tf.math.divide_no_nan(1.0,T2S) - tf.math.divide_no_nan(1.0,T2)) - tf.math.divide_no_nan(t,T2) )
    return output

FID_Layer = layers.Lambda(simulateSignal_for_FID, name = 'FID')([dense_layer_3a,dense_layer_3c])
Echo_Peak_rise_layer = layers.Lambda(simulateSignal_for_Echo_Peak_rise, name = 'SE_rise')([dense_layer_3a,dense_layer_3b,dense_layer_3c])
Echo_Peak_fall_layer = layers.Lambda(simulateSignal_for_Echo_Peak_fall, name = 'SE_fall')([dense_layer_3a,dense_layer_3b,dense_layer_3c])
output_layer = layers.Concatenate(name = 'Output_layer')([FID_Layer,Echo_Peak_rise_layer,Echo_Peak_fall_layer])


model = keras.Model(inputs=input_layer,outputs=output_layer,name="Lambda_model")
model.summary()
keras.utils.plot_model(model, show_shapes=True)


# %% Train full model
model.compile(
    loss=keras.losses.MeanAbsolutePercentageError(),
    optimizer='adam',
    metrics=["accuracy"],
)

model_wrapper = lm.ModelWrapper(
    tf.keras.models.clone_model(model))

model_wrapper.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
    loss=lm.ReducedOutputsMeanSquaredError())

# %%

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_07_15-1330')
]

history = model.fit(signal_train_noise, signal_train_noise, batch_size=50, epochs=100, validation_split=0.2, callbacks=my_callbacks)
test_scores = model.evaluate(signal_test_noise, signal_test, verbose=2)
test_scores = model.evaluate(signal_test_noise, signal_test_noise, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
test_scores_params = model_params.evaluate(signal_test_noise, y_test, verbose=2)

model_params.save("Model_Params_after.h5")
model.save("Model_Full.h5")


# %%
print("Train using Levenberg-Marquardt")
#train_dataset = tf.data.Dataset.from_tensor_slices((signal_train_noise, signal_train_noise))
t2_start = time.perf_counter()
model_wrapper.fit(signal_train_noise, signal_train_noise, batch_size = 500, epochs=100)
t2_stop = time.perf_counter()
print("Elapsed time: ", t2_stop - t2_start)

#%% Look at predictions


for Number in range(5):
    p = model.predict(signal_test)
    p_params = model_params.predict(signal_test)
    #print(p[Number,:])
    #print(p.shape)
    #print(signal_test[Number,:])
    plt.figure()
    plt.plot(t,signal_test[Number,:],'o-')
    plt.plot(t,signal_test_noise[Number,:],'o')
    plt.plot(t,p[Number,:],'o--')
    plt.legend(['Truth S0: {:.0f} T2: {:.1f} T2*: {:.1f}'.format(y_test[Number,0],y_test[Number,1],y_test[Number,2]),
                'Input with added noise',
                'Pred S0: {:.0f} T2: {:.1f} T2*: {:.1f}'.format(p_params[Number,0],p_params[Number,1],p_params[Number,2])])
    plt.title(str(Number))

    #print(['S0', 'T2', 'T2S'])
    #print(p_params[Number,:])
    #print(y_test[Number,:])
