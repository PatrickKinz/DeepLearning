import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.models import Model

# Create the red, green and blue channels:

red   = np.array([1]*9).reshape((3,3))
green = np.array([100]*9).reshape((3,3))
blue  = np.array([10000]*9).reshape((3,3))

# Stack the channels to form an RGB image:

img = np.stack([red, green, blue], axis=-1)
img = np.expand_dims(img, axis=0)

# Create a model that just does a Conv2D convolution:
# %%
inputs2D = Input((3,3,3))
conv2D = Conv2D(filters=1,
              strides=1,
              padding='valid',
              activation='relu',
              kernel_size=2,
              kernel_initializer='ones',
              bias_initializer='zeros', )(inputs2D)
model2Dconv = Model(inputs2D,conv2D)

model2Dconv.summary()
"""
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 3, 3, 3)]         0
_________________________________________________________________
conv2d (Conv2D)              (None, 2, 2, 1)           13
=================================================================
Total params: 13
Trainable params: 13
Non-trainable params: 0
"""
model2Dconv.predict(img)
"""
 array([[[[40404.],
          [40404.]],

         [[40404.],
          [40404.]]]], dtype=float32)
"""
# %%
depth2D = DepthwiseConv2D(
              strides=1,
              padding='valid',
              activation='relu',
              kernel_size=2,
              depthwise_initializer='ones',
              bias_initializer='zeros', )(inputs2D)
model2Ddepth = Model(inputs2D,depth2D)

model2Ddepth.summary()

"""
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 3, 3, 3)]         0
_________________________________________________________________
depthwise_conv2d (DepthwiseC (None, 2, 2, 3)           15
=================================================================
Total params: 15
Trainable params: 15
Non-trainable params: 0
"""
model2Ddepth.predict(img)

"""
array([[[[4.e+00, 4.e+02, 4.e+04],
         [4.e+00, 4.e+02, 4.e+04]],

        [[4.e+00, 4.e+02, 4.e+04],
         [4.e+00, 4.e+02, 4.e+04]]]], dtype=float32)
"""
# %%
separable2D = SeparableConv2D(filters=1,
              strides=1,
              padding='valid',
              activation='relu',
              kernel_size=2,
              depthwise_initializer='ones',
              pointwise_initializer='ones',
              bias_initializer='zeros', )(inputs2D)
model2Dseparable = Model(inputs2D,separable2D)

model2Dseparable.summary()

"""
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 3, 3, 3)]         0
_________________________________________________________________
separable_conv2d (SeparableC (None, 2, 2, 1)           16
=================================================================
Total params: 16
Trainable params: 16
Non-trainable params: 0
"""
model2Dseparable.predict(img)

"""
array([[[[40404.],
         [40404.]],

        [[40404.],
         [40404.]]]], dtype=float32)
"""
# %%

inputs3D = Input((3,3,3,1))

conv3D = Conv3D(filters=1,
              strides=1,
              padding='valid',
              activation='relu',
              kernel_size=[2,2,2],
              kernel_initializer='ones',
              bias_initializer='zeros', )(inputs3D)
model3D = Model(inputs3D,conv3D)

model3D.summary()
"""
Layer (type)                 Output Shape              Param #
=================================================================
input_4 (InputLayer)         [(None, 3, 3, 3, 1)]      0
_________________________________________________________________
conv3d_4 (Conv3D)            (None, 2, 2, 2, 1)        9
=================================================================
Total params: 9
Trainable params: 9
Non-trainable params: 0
"""
model3D.predict(img)
"""
array([[[[[  404.],
          [40400.]],

         [[  404.],
          [40400.]]],


        [[[  404.],
          [40400.]],

         [[  404.],
          [40400.]]]]], dtype=float32)
"""
# %%
inputs3D = Input((3,3,3,1))

conv3D = Conv3D(filters=5,
              strides=1,
              padding='valid',
              activation='relu',
              kernel_size=[2,2,3],
              kernel_initializer='ones',
              bias_initializer='zeros', )(inputs3D)
model3D = Model(inputs3D,conv3D)

model3D.summary()
"""
Layer (type)                 Output Shape              Param #
=================================================================
input_3 (InputLayer)         [(None, 3, 3, 3, 1)]      0
_________________________________________________________________
conv3d_3 (Conv3D)            (None, 2, 2, 1, 1)        13
=================================================================
Total params: 13
Trainable params: 13
Non-trainable params: 0
"""
model3D.predict(img)
"""
array([[[[[40404.]],

         [[40404.]]],


        [[[40404.]],

         [[40404.]]]]], dtype=float32)
"""
