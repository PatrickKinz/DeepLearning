from skimage import io
from tensorflow import keras
import tensorflow as tf
import numpy as np

class My_Params_Generator(keras.utils.Sequence) :

  def __init__(self, image_filenames, batch_size) :
    self.image_filenames = image_filenames
    #self.labels = labels
    self.batch_size = batch_size


  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)


  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    #batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    qBOLD  = np.array([io.imread('../Brain_Phantom/Patches_no_air_big/qBOLD/qBOLD_' + str(file_name)) for file_name in batch_x])
    qBOLD = tf.convert_to_tensor(np.moveaxis(qBOLD,1,-1),dtype=tf.float32)
    #could add noise to qBOlD and QSM here
    QSM    = np.array([io.imread('../Brain_Phantom/Patches_no_air_big/QSM/QSM_' + str(file_name)) for file_name in batch_x])
    QSM = tf.convert_to_tensor(np.moveaxis(QSM,1,-1),dtype=tf.float32)
    Params = np.array([io.imread('../Brain_Phantom/Patches_no_air_big/Params/Params_' + str(file_name)) for file_name in batch_x])
    Params = np.moveaxis(Params,1,-1)
    S0 = tf.convert_to_tensor(Params[:,:,:,0],dtype=tf.float32)
    R2 = tf.convert_to_tensor(Params[:,:,:,1],dtype=tf.float32)
    Y = tf.convert_to_tensor(Params[:,:,:,2],dtype=tf.float32)
    nu = tf.convert_to_tensor(Params[:,:,:,3],dtype=tf.float32)
    chi_nb = tf.convert_to_tensor(Params[:,:,:,4],dtype=tf.float32)
    return [qBOLD, QSM], [S0,R2,Y,nu,chi_nb]


class My_Signal_Generator(keras.utils.Sequence) :

  def __init__(self, image_filenames, batch_size) :
    self.image_filenames = image_filenames
    #self.labels = labels
    self.batch_size = batch_size


  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)


  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]

    qBOLD  = np.array([io.imread('/qBOLD/qBOLD_' + str(file_name)) for file_name in batch_x])
    #could add noise to qBOlD and QSM here
    QSM    = np.array([io.imread('/QSM/QSM_' + str(file_name)) for file_name in batch_x])

    return [qBOLD, QSM], [qBOLD, QSM]
