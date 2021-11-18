import tensorflow as tf
import numpy as np
#%%
a= tf.keras.layers.Input((30,30))
b= tf.keras.layers.Input((30,30))
c= tf.keras.layers.Input((30,30))
d= tf.keras.layers.Input((30,30))
e= tf.keras.layers.Input((30,30))
a.shape
print(a.shape[1])


a_flat=tf.keras.layers.Flatten()(a)
b_flat=tf.keras.layers.Flatten()(b)
c_flat=tf.keras.layers.Flatten()(c)
d_flat=tf.keras.layers.Flatten()(d)
e_flat=tf.keras.layers.Flatten()(e)
a_flat.shape

n_grid = 20
c_start = tf.zeros([10,c_flat.shape[1],n_grid])                             #shape (n_batch,n_voxel,n_grid)
c_start.shape
c_stop  = tf.ones([10,c_flat.shape[1],n_grid])                             #shape (n_batch,n_voxel,n_grid)
c_plane = tf.linspace(c_start,c_stop,n_grid,axis=-2)     #shape (n_batch,n_voxel,n_grid,n_grid) Y varied along first n_grid
c_plane.shape
d_start = tf.zeros([10,d_flat.shape[1],n_grid])                             #shape (n_batch,n_voxel,n_grid)
d_start.shape
d_stop  = tf.ones([10,d_flat.shape[1],n_grid])                             #shape (n_batch,n_voxel,n_grid)
d_plane = tf.linspace(d_start,d_stop,n_grid,axis=-1)     #shape (n_batch,n_voxel,n_grid,n_grid) Y varied along first n_grid
d_plane.shape


test= tf.zeros_like(c_flat)
test.shape
test_start=tf.linspace(test,test,n_grid)
test_start.shape


test2 = tf.zeros([5])
test2.shape
test2.shape[0]

test3=tf.broadcast_to(tf.expand_dims(test2,-1),[5,2])
test3.shape


test4 = tf.ones([5,1])
test5 = tf.ones([1,3])
test6 = test4*test5
test6.shape
[test6.shape[0],test6.shape[1],6]


a=tf.constant([1,2,3,4,5],dtype=tf.float32)
a=tf.expand_dims(a,-1)
a=tf.expand_dims(a,-1)
a.shape
b=tf.ones([5,2,3])
b.shape
c=a*b
c.shape
c
