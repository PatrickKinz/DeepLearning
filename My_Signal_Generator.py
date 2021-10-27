from skimage import io

class My_Custom_Generator(keras.utils.Sequence) :

  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size


  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)


  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    #batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    qBOLD  = np.array([io.imread('/qBOLD/qBOLD_' + str(file_name)) for file_name in batch_x])
    #could add noise to qBOlD and QSM here
    QSM    = np.array([io.imread('/QSM/QSM_' + str(file_name)) for file_name in batch_x])
    Params = np.array([io.imread('/Params/Params_' + str(file_name)) for file_name in batch_x])

    return [qBOLD, QSM], [Params[:,:,:,0],Params[:,:,:,1],Params[:,:,:,2],Params[:,:,:,3],Params[:,:,:,4]]
