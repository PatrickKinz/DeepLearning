# %% import modules
import tensorflow as tf
tf.config.list_physical_devices('GPU')
import numpy as np
from numpy.random import rand, randn, random
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# %% Create Data

def f_hyper(x):
    '''
    Write hypergeometric function as taylor order 10 for beginning and as x-1 for larger numbers
    Exakt equation: hypergeom(-0.5,[0.75,1.25],-9/16*x.^2)-1
    (Intersection>x)*taylor + (x>=Intersection)*(x-1)
    taylor = - (81*x^8)/10890880 + (27*x^6)/80080 - (3*x^4)/280 + (3*x^2)/10
    Intersection at approx x = 3.72395
    '''
    Intersection = 3.72395
    a = (Intersection>x)*( -81./10890880*pow(x,8) +27./80080*pow(x,6) -3./280*pow(x,4) +3./10*pow(x,2) )
    b = (x>=Intersection)*(x-1)
    return a + b

def f_qBOLD(S0, R2, Y, nu, chi_nb, t):
    output = np.zeros((len(S0),len(t)))
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
    #print('chi_ba', chi_ba)
    #CF = gamma *B0
    gamma = 267.513 #MHz/T
    B0 = 3 #T
    delta_chi0 = 4*np.pi*0.273 #in ppm
    delta_chi0*Hct
    dw = 1./3 * gamma * B0* (Hct * delta_chi0 * (1-Y) + chi_ba - chi_nb )
    #print('dw', dw, dw.shape)
    #FID
    for i in range(6):
        output[:,i] = S0 * np.exp(-R2*t[i] -nu * f_hyper(dw *     t[i] ) )
    #SE rise
    for i in range(6,13):
        output[:,i] = S0 * np.exp(-R2*t[i] -nu * f_hyper(dw * (TE-t[i]) ) )
    #SE fall
    for i in range(13,len(t)):
        output[:,i] = S0 * np.exp(-R2*t[i] -nu * f_hyper(dw * (t[i]-TE) ) )
    return output

def f_QSM(Y, nu, chi_nb ):
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

    a = (chi_ba/alpha +psi_Hb*delta_chi_Hb * ((1-(1-alpha)*SaO2)/alpha - Y) )*nu
    b = (1 - nu/alpha) * chi_nb
    return np.array([a + b]).T

def convert_params_GM(y):
    """ for normal distributed params in GM """
    S0 = y[0]
    R2 = 12 + 6*y[1]
    Y = 0.6 + 0.15*y[2]
    nu = 0.04 + 0.02*y[3]
    chi_nb = 0.1  + 0.15*y[4]
    return S0,R2,Y,nu,chi_nb

def convert_params_WM(y):
    """ for normal distributed params in WM """
    S0 = y[0]
    R2 = 17 + 4*y[1]
    Y = 0.6 + 0.15*y[2]
    nu = 0.02 + 0.02*y[3]
    chi_nb = -0.1  + 0.15*y[4]
    return S0,R2,Y,nu,chi_nb

def convert_params(y):
    """ for uniform distributed params """
    S0 = y[0]   #S0     = 1000 + 200 * randn(N).T
    R2 = (30-1) * y[1] + 1
    SaO2 = 0.98
    Y  = (SaO2 - 0.01) * y[2] + 0.01
    nu = (0.1 - 0.001) * y[3] + 0.001
    chi_nb = ( 0.1-(-0.1) ) * y[4] - 0.1
    return S0,R2,Y,nu,chi_nb

def createData1D(N,t):
    """
    #Different means for Gray Matter and White Matter
    N  = int(N/2)
    b1 = randn(N)
    b2 = randn(N)
    b  = np.concatenate((b1,b2))
    R2 = np.concatenate( (12+6*b1 , 17+4*b2) ).T
    #print('R2', R2, R2.shape)
    c = randn(2*N)
    Y = 0.6 + 0.15*c.T
    #print('Y', Y, Y.shape)
    d1 = randn(N)
    d2 = randn(N)
    d  = np.concatenate((d1,d2))
    nu = np.concatenate((0.04 + 0.02*d1, 0.02 + 0.02*d2) ).T
    #print('nu', nu, nu.shape)
    e1 = randn(N)
    e2 = randn(N)
    e  = np.concatenate((e1,e2))
    chi_nb = np.concatenate((0.1  + 0.15*e1, -0.1 + 0.15*e2) ).T
    #print('chi_nb', chi_nb, chi_nb.shape)
    a  = 0.5 * np.ones(R2.shape) #1000 + 200 * randn(2*N).T
    S0 = a
    #print('S0',S0,S0.shape)

    Need to remove unphysical values.
    0.1   < R2  < 100
    0.01  < Y   < Sa02=0.98
    0.001 < nu  < alpha(=0.77)-0.01
    -inf  < chi < inf
    """
    b=random(N)
    a=0.5*np.ones(b.shape)
    c=random(N)
    d=random(N)
    e=random(N)
    S0 = a   #S0     = 1000 + 200 * randn(N).T
    R2 = (30-1) * b + 1
    SaO2 = 0.98
    Y  = (SaO2 - 0.01) * c + 0.01
    nu = (0.1 - 0.001) * d + 0.001
    chi_nb = ( 0.1-(-0.1) ) * e - 0.1

    signal = f_qBOLD(S0,R2,Y,nu,chi_nb,t)
    #print('signal', signal.shape)
    QSM = f_QSM(Y,nu,chi_nb)
    #print('QSM',QSM.shape)
    data = np.append(signal,QSM,axis=1)
    targets = np.array([a,b,c,d,e]).T
    return data, targets
# %%
t=np.array([3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48])/1000
data, targets = createData1D(10,t)
data
data.shape
targets.shape

#for i in range(4):
#    plt.figure()
#    plt.plot(t,targets[i,0]*np.exp(-targets[i,1]*t),'o-')
#    plt.plot(t,data[i,:16].T, 'o-')
    #plt.txt(targets[i,:])
plt.figure()
plt.plot(t,data[:5,:16].T, 'o-')
plt.figure()
plt.plot(t,data[5:,:16].T, 'o-')


# %%


N_train = 10000000
N_test = 10000

signal_train, y_train = createData1D(N_train,t)
signal_test, y_test = createData1D(N_test,t)
print(signal_train.shape)
print(y_train.shape)

plt.plot(t,np.transpose(signal_train[:10,:16]), 'o-')



y_train[6]
convert_params_GM(y_train[6])
y_train[7]
convert_params_GM(y_train[7])
# %% Add noise to data

def addNoise(input,Spread,Offset):
    output = np.multiply(input, 1 + Spread*randn(input.size).reshape(input.shape)) + Offset*rand(input.size).reshape(input.shape)
    return output

signal_train_noise = addNoise(signal_train, 0.02, 0)
print(signal_train_noise.shape)
signal_test_noise = addNoise(signal_test, 0.02, 0)

plt.plot(t,np.transpose(signal_train_noise[:5,:16]), 'o-')

# %% Network

input_layer = keras.Input(shape=(17,), name = 'Input_layer')
input_layer.shape
input_layer.dtype

dense_layer_1 = layers.Dense(12, activation="relu", name = 'Dense_1')(input_layer)
dense_layer_1.shape
dense_layer_2 =layers.Dense(24,activation="relu", name = 'Dense_2')(dense_layer_1)
dense_layer_3 =layers.Dense(12,activation="relu", name = 'Dense_3')(dense_layer_2)
dense_layer_3a = layers.Dense(1,activation="sigmoid", name = 'Dense_3a_S0')(dense_layer_3) # 3 outputs for S0, T2 and T2S
dense_layer_3b = layers.Dense(1,activation="sigmoid", name = 'Dense_3b_R2')(dense_layer_3) # 3 outputs for S0, T2 and T2S
dense_layer_3c = layers.Dense(1,activation="sigmoid", name = 'Dense_3c_Y')(dense_layer_3) # 3 outputs for S0, T2 and T2S
dense_layer_3d = layers.Dense(1,activation="sigmoid", name = 'Dense_3d_nu')(dense_layer_3) # 3 outputs for S0, T2 and T2S
dense_layer_3e = layers.Dense(1,activation="sigmoid", name = 'Dense_3e_chi_nb')(dense_layer_3) # 3 outputs for S0, T2 and T2S

#before_lambda_model = keras.Model(input_layer, dense_layer_3, name="before_lambda_model")

Params_Layer = layers.Concatenate(name = 'Output_Params')([dense_layer_3a,dense_layer_3b,dense_layer_3c,dense_layer_3d,dense_layer_3e])
model_params = keras.Model(inputs=input_layer,outputs=Params_Layer,name="Params_model")
model_params.summary()
keras.utils.plot_model(model_params, show_shapes=True)

# %% Train Params model
model_params.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer='adam',
    metrics=[tf.keras.metrics.MeanSquaredError()],
)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_08_06-1415')
]



history = model_params.fit(signal_train, y_train, batch_size=1000, epochs=500, validation_split=0.2, callbacks=my_callbacks)
test_scores = model_params.evaluate(signal_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])



p = model_params.predict(signal_test)
Number = 2
#                        S0          R2        Y           nu         chi_nb
print(y_test[Number,:])
print(p[Number     ,:])

print(convert_params(y_test[Number,:]))
print(convert_params(p[Number,:]))

model_params.save("Model_Params_before.h5")





# %%

def f_hyper(x):
    '''
    Write hypergeometric function as taylor order 10 for beginning and as x-1 for larger numbers
    Exakt equation: hypergeom(-0.5,[0.75,1.25],-9/16*x.^2)-1
    (Intersection>x)*taylor + (x>=Intersection)*(x-1)
    taylor = - (81*x^8)/10890880 + (27*x^6)/80080 - (3*x^4)/280 + (3*x^2)/10
    Intersection at approx x = 3.72395
    '''
    Intersection = 3.72395
    mask_a = tf.math.greater(Intersection,x)
    mask_b = ~mask_a
    mask_a = tf.cast(mask_a, tf.float32)
    mask_b = tf.cast(mask_b, tf.float32)
    #coefficients             x8, x7,       x6, x5,     x4, x3,  x2, x1, x0
    coefficients = [-81./10890880,  0., 27./80080,  0., -3./280,  0., 0.3,  0.,  0.]
    a = tf.math.multiply(tf.math.polyval(coefficients,x), mask_a)
    #print(a)
    b = tf.math.multiply(x-1, mask_b)
    #print(b)
    return a + b

t=np.array([3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48], dtype=np.float32)/1000
tc=8./1000
t.dtype
plt.plot(t,f_hyper(t/tc),'o-')




def f_qBOLD_tensor(tensor):
    TE = 40
    t=np.array([3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48], dtype=np.float32)/1000
    t_hyper=np.array([3,6,9,12,15,18,TE-21,TE-24,TE-27,TE-30,TE-33,TE-36,TE-39,42-TE,45-TE,48-TE], dtype=np.float32)/1000
    S0     = tensor[0]
    R2     = tensor[1]
    Y      = tensor[2]
    nu     = tensor[3]
    chi_nb = tensor[4]
    Hct = 0.357
    # Blood Hb volume fraction
    psi_Hb = Hct*0.34/1.335
    # Susceptibility of oxyhemoglobin in ppm
    chi_oHb = -0.813
    # Susceptibility of plasma in ppm
    chi_p = -0.0377
    # Susceptibility of fully oxygenated blood in ppm
    chi_ba = psi_Hb*chi_oHb + (1-psi_Hb)*chi_p
    #print('chi_ba', chi_ba)
    #CF = gamma *B0
    gamma = 267.513 #MHz/T
    B0 = 3 #T
    delta_chi0 = 4*np.pi*0.273 #in ppm
    dw = 1./3 * gamma * B0* (Hct * delta_chi0 * (1-Y) + chi_ba - chi_nb )
    return S0 *tf.math.exp(-R2 * t - nu *f_hyper(dw*t_hyper) )


def decode_S0(tensor):
    return tensor
def decode_R2(tensor):
    return (30-1) * tensor + 1
def decode_Y(tensor):
    SaO2 = 0.98
    return (SaO2 - 0.01)  * tensor + 0.01
def decode_nu(tensor):
    return (0.1 - 0.001)  * tensor + 0.001
def decode_chi_nb(tensor):
    return ( 0.1-(-0.1) ) * tensor - 0.1

S0_layer     = layers.Lambda(decode_S0, name= 'S0')(dense_layer_3a)
R2_layer     = layers.Lambda(decode_R2, name= 'R2')(dense_layer_3b)
Y_layer      = layers.Lambda(decode_Y,  name=  'Y')(dense_layer_3c)
nu_layer     = layers.Lambda(decode_nu, name= 'nu')(dense_layer_3d)
chi_nb_layer = layers.Lambda(decode_chi_nb, name= 'chi_nb')(dense_layer_3e)


Input= [800,17,0.6,0.05,0.05]
f_qBOLD_tensor(Input)
plt.plot(t,f_qBOLD_tensor(Input),'o-')


def f_QSM_tensor(tensor):
    Y=tensor[0]
    nu=tensor[1]
    chi_nb=tensor[2]
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
    a = (chi_ba/alpha +psi_Hb*delta_chi_Hb * ((1-(1-alpha)*SaO2)/alpha - Y) )*nu
    b = (1 - nu/alpha) * chi_nb
    return a + b


qBOLD_layer  = layers.Lambda(f_qBOLD_tensor, name= 'qBOLD')([S0_layer,R2_layer,Y_layer,nu_layer,chi_nb_layer])
QSM_layer    = layers.Lambda(f_QSM_tensor, name= 'QSM')([Y_layer,nu_layer,chi_nb_layer])
output_layer = layers.Concatenate(name = 'Output_layer')([qBOLD_layer,QSM_layer])

model = keras.Model(inputs=input_layer,outputs=output_layer,name="Lambda_model")
model.summary()
keras.utils.plot_model(model, show_shapes=True)


# %% Train full model
model.compile(
    loss=keras.losses.MeanAbsolutePercentageError(),
    optimizer='adam',
    metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs/2021_08_06-1600')
]

history = model.fit(signal_train, signal_train, batch_size=500, epochs=100, validation_split=0.2, callbacks=my_callbacks)
test_scores = model.evaluate(signal_test, signal_test, verbose=2)
#test_scores = model.evaluate(signal_test_noise, signal_test_noise, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
test_scores_params = model_params.evaluate(signal_test, y_test, verbose=2)

model_params.save("Model_Params_after.h5")
model.save("Model_Full.h5")
#%% Look at predictions


p = model.predict(signal_test)
p_params = model_params.predict(signal_test)
for Number in range(10):
    #print(p[Number,:])
    #print(p.shape)
    #print(signal_test[Number,:])
    plt.figure()
    plt.plot(t,signal_test[Number,:16],'o-')
#    plt.plot(t,signal_test_noise[Number,:],'o')
    plt.plot(t,p[Number,:16],'x--')
    plt.legend(['T: S0:{:.2f} R2:{:.2f} Y:{:.2f} nu:{:.2f} chi:{:.3f}'.format(y_test[Number,0],y_test[Number,1],y_test[Number,2],y_test[Number,3],y_test[Number,4]),
#                'Input with added noise',
                'P: S0:{:.2f} R2:{:.2f} Y:{:.2f} nu:{:.2f} chi:{:.3f}'.format(p_params[Number,0],p_params[Number,1],p_params[Number,2],p_params[Number,3],p_params[Number,4])])
    plt.title(str(Number))

    #print(['S0', 'T2', 'T2S'])
    #print(p_params[Number,:])
    #print(y_test[Number,:])
