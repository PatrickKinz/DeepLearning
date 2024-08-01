import tensorflow as tf
tf.config.list_physical_devices('GPU')
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

inputs = keras.Input(shape=(784,))
inputs.shape
inputs.dtype

dense = layers.Dense(64, activation="relu")
x=dense(inputs)

x=layers.Dense(64,activation="relu")(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs,outputs=outputs,name="mnist_model")
model.summary()
keras.utils.plot_model(model)
keras.utils.plot_model(model, show_shapes=True)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

x_train.shape

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
