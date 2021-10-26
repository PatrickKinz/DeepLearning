class My_Custom_Generator(keras.utils.Sequence) :

  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size


  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)


  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    return np.array([
            resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
               for file_name in batch_x])/255.0, np.array(batch_y)

#%%
from skimage import io
a=io.imread("../Brain_Phantom/Patches_no_air/Params/Params_000000.TIF")
a=io.imread("../Brain_Phantom/Patches_no_air/qBOLD/qBOLD_000000.TIF")
a.shape