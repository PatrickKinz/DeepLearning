import numpy as np
import matplotlib.pyplot as plt

# to make DLIPR available put 'software community/dlipr' in your ~/.profile
#import sys
#sys.path.append("C:/Users/patri/OneDrive/Uni/ss18/Deep Learning/dlipr-master/dlipr-master/")

#import dlipr

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as KTF
from tensorflow.keras.layers import Dense, Convolution2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
# ----------------------------------------------
# Data
# ----------------------------------------------

#data = dlipr.mnist.load_data()

# plot some examples
#data.plot_examples(fname='examples.png')
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
X_train = x_train.reshape(60000, 28, 28, 1)
X_test = x_test.reshape(10000, 28, 28, 1)

# reshape the image matrices
#X_train = data.train_images.reshape(data.train_images.shape[0], 28, 28, 1)
#X_test = data.test_images.reshape(data.test_images.shape[0], 28, 28, 1)
#print('%i training samples' % X_train.shape[0])
#print('%i test samples' % X_test.shape[0])

# Preprocess data : convert integer RGB values (0-255) to float values (0-1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# convert class labels to one-hot encodings
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)
#Y_train = to_categorical(data.train_labels, 10)
#Y_test = to_categorical(data.test_labels, 10)


# ----------------------------------------------
# Model and training
# ----------------------------------------------

n=30 #erreicht 98% nur manchmal gerade so. Mehr filter sollten helfen

model = Sequential( [ Convolution2D(n,                input_shape=(28,28,1),     kernel_size=(3,3),strides=(1,1),padding='same',dilation_rate=1,activation='relu'), Convolution2D(n,kernel_size=(3,3),strides=(1,1),padding='same',dilation_rate=1,activation='relu'), Convolution2D(2*n,kernel_size=(3,3),strides=(1,1),padding='same',dilation_rate=2,activation='relu'), Convolution2D(4*n,kernel_size=(3,3),strides=(1,1),padding='same',dilation_rate=1,activation='relu'),GlobalAveragePooling2D(data_format=None), Dense(10, activation='softmax') ]  )
# Set up a convolutional network with at least 3 convolutional layers
# and train the network to at least 98% test accuracy
#
# model.compile()
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=1e-3),
    metrics=['accuracy'])

# results = model.fit(X_train, Y_train,
#     validation_split=0.1,  # split off 10% of training data for validation
#     )
results = model.fit(
    X_train, Y_train,
    batch_size=100,
    epochs=10,
    verbose=2,
    validation_split=0.1,  # split off 10% of training data for validation
    callbacks=[])

#test accuracy
[loss_test, test_acc] = model.evaluate(X_test, Y_test, verbose=0)
print('test accuracy: ', test_acc)

model.save("mnist_model.h5") # save your model

fig1, (ax1,ax2) = plt.subplots(2,1)

# plot data
ax1.plot(results.history['accuracy'],label='acc')
ax1.plot(results.history['val_accuracy'],label='val_acc')
ax1.set(xlabel='epoch', ylabel='accuracy')
ax1.legend()

ax2.plot(results.history['loss'],label='loss')
ax2.plot(results.history['val_loss'],label='val_loss')
ax2.set(xlabel='epoch', ylabel='loss')
ax2.legend()

plt.tight_layout()
fig1.savefig("training_history.png")
fig1


#%%
# model = load_model("mnist_model.h5")  # load your trained model -> you only have to train the model once!
# model.summary()
# ----------------------------------------------
# Visualization
# ----------------------------------------------


def deprocess(x):
    '''Use this function before plotting the visualized feature map'''
    # standard normalize the tensor
    x -= x.mean()
    x /= (x.std() + KTF.epsilon())
    x *= 0.1

    # clip values [0, 1] and convert back to RGB
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    '''Normalize the gradients via the l2 norm'''
    return x/(KTF.sqrt(KTF.mean(KTF.square(x))) + KTF.epsilon())


model_input = model.inputs  # Define model inputs. Add [KTF.learning_phase()] flag when using dropout before the layers you want to visualize

layer_dict = dict([(layer.name, layer) for layer in model.layers[:]])
layer_names = []  # build array of layer_names you would like to visualize
gradient_updates = 50
step_size = 1.


for layer_name in layer_names:
    layer_output = layer_dict[layer_name].output
    filter = []
    for filter_index in range(layer_output.shape[-1].value):

        # define a scalar loss using Keras functions.
        # remember: You would like to maximize the activations in the respective feature map!
        loss = 0

        # start from uniform distributed noise images and choose one specific feature map
        # Then calculate the gradient of the loss w.r.t. the input
        # Afterwards add the calculated gradients to your start image (gradient ascent step) and repeat the procedure.

        # You can calculate the gradients using the following expressions:
        # gradients = KTF.gradients(loss, model_input)[0]
        # gradients = normalize(gradients)
        # iterate = KTF.function(model_input, [loss, gradients])

        # don't forget to use deprocess() before you plot the feature map
        # finally plot the visualized filter and comment on the patterns you observe
        # Did the visualization work for all feature maps?
