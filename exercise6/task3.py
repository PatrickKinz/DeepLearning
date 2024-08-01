from densenet import DenseNet

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("C:/Users/patri/OneDrive/Uni/ss18/Deep Learning/dlipr-master/dlipr-master/")

import dlipr
import keras
import time
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import plot_model


start=time.time()

# ---------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------
data = dlipr.cifar.load_cifar10()

# plot some example images
dlipr.utils.plot_examples(data, num_examples=5, fname='examples.png')

print(data.train_images.shape)
print(data.train_labels.shape)
print(data.test_images.shape)
print(data.test_labels.shape)

# preprocess the data in a suitable way

# convert integer RGB values (0-255) to float values (0-1)
X_train = data.train_images.astype('float32') / 255
X_test = data.test_images.astype('float32') / 255

# convert class labels to one-hot encodings
Y_train = to_categorical(data.train_labels, 10)
Y_test = to_categorical(data.test_labels, 10)
print(Y_train.shape)




# ----------------------------------------------------------
# Model and training
# ----------------------------------------------------------


model = DenseNet(input_shape=(32, 32, 3),
        num_classes=10,
        dense=3, #orginal default values from the densenet class
        layers=12
        )
#model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=1e-3),
    metrics=['accuracy'])

results = model.fit(
    X_train, Y_train,
    batch_size=100,
    epochs=10,
    verbose=2,
    validation_split=0.1,  # split off 10% of training data for validation
    callbacks=[])

"""#####  1.2 Plot confusion matrix            ########"""
#test accuracy
[loss_test, test_acc] = model.evaluate(X_test, Y_test, verbose=0)
print('test accuracy: ', test_acc)

# predicted probabilities for the test set
Yp = model.predict(X_test)
#print(Yp.shape)
yp = np.argmax(Yp, axis=1)


# plot the confusion matrix
dlipr.utils.plot_confusion(yp, data.test_labels, data.classes,
                           fname='confusion.png')


end=time.time()
m, s = divmod(end-start, 60)
print('runtime:',m,"min",s,"sec")