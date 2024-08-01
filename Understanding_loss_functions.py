import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import numpy as np
"""
tf.keras.losses.MeanAbsolutePercentageError(
    reduction=losses_utils.ReductionV2.AUTO,
    name='mean_absolute_percentage_error'
)

loss = 100 * abs(y_true - y_pred) / y_true
"""
#%%
y_true = [[2., 1.], [2., 3.]]
np.shape(y_true)
y_true[0]
y_pred = [[1., 1.], [1., 0.]]
# Using 'auto'/'sum_over_batch_size' reduction type.
mape = tf.keras.losses.MeanAbsolutePercentageError()
mape(y_true, y_pred).numpy()

#%%
# Calling with 'sample_weight'.
mape(y_true, y_pred, sample_weight=[1, 0]).numpy()

#%%
# Using 'sum' reduction type.
mape = tf.keras.losses.MeanAbsolutePercentageError(
    reduction=tf.keras.losses.Reduction.SUM)
mape(y_true, y_pred).numpy()



#%%
# Using 'none' reduction type.
mape = tf.keras.losses.MeanAbsolutePercentageError(
    reduction=tf.keras.losses.Reduction.NONE)
mape(y_true, y_pred).numpy()
