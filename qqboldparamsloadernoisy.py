import logging
import os
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from skimage import io


from dataloader import DataLoader
import config as cfg

# configure logger
logger = logging.getLogger(__name__)

class QQBoldParamsLoader(DataLoader):
    """
    seed : int, optional
        set a fixed seed for the loader, by default 42
    mode : has no effect, should not be Apply
    name : str, optional
        the name of the loader, by default 'reader'
    """

    def __init__(
        self,
        mode=None,
        seed=42,
        name="reader",
        shuffle=None,
        sample_buffer_size=None,
        **kwargs,
    ):

        super().__init__(
            mode=mode,
            seed=seed,
            name=name,
            shuffle=shuffle,
            sample_buffer_size=sample_buffer_size,
            **kwargs,
        )
        # set the capacity
        if sample_buffer_size is None:
            self.sample_buffer_size = cfg.batch_capacity_train


    def __call__(
        self,
        file_list: List[str],
        batch_size: Optional[int] = None,
        n_epochs=50,
        read_threads=1,
        **kwargs,
    ) -> tf.data.Dataset:
        """Create the tensorflow dataset when calling the data loader

        Parameters
        ----------
        file_list : List[str]
            The files that should be loaded
        batch_size : int
            The batch size, if None, cfg.train_batch_size will be used, by default None
        n_epochs : int, optional
            The number of epochs. Each file will be used once per epoch, by default 50
        read_threads : int, optional
            The number of read threads, by default 1

        Returns
        -------
        tf.data.Dataset
            The tensorflow dataset, all files will be shuffled each epoch, then the
            samples will be interleaved and the dataset is shuffled again with a
            buffer 3 * number of files
        """
        return super().__call__(
            file_list, batch_size, n_epochs=n_epochs, read_threads=read_threads
        )

    def _set_up_shapes_and_types(self):
        """
        sets:
        - dtypes
        - dshapes

        """
        # dtypes and dshapes are defined in the base class
        # pylint: disable=attribute-defined-outside-init

        if self.mode is self.MODES.TRAIN or self.mode is self.MODES.VALIDATE:
            self.dtypes = [tf.float32,tf.float32, tf.float32,tf.float32,tf.float32,tf.float32,tf.float32]
            self.dshapes = [
                np.array([30,30,16]),np.array([30,30,1]),
                np.array([30,30]),np.array([30,30]),np.array([30,30]),np.array([30,30]),np.array([30,30]),
            ]
        else:
            raise ValueError(f"Not allowed mode {self.mode}")

        #self.data_rank = len(self.dshapes[0])

        #assert self.data_rank in [3, 4], "The rank should be 3 or 4."

    def _read_file_and_return_numpy_samples(self, file_name_queue: bytes):
        """Helper function getting the actual samples

        Parameters
        ----------
        file_name_queue : bytes
            The filename

        Returns
        -------
        np.array, np.array
            The samples and labels
        """
        #data_img, label_img = self._load_file(file_name_queue)
        #samples, labels = self._get_samples_from_volume(data_img, label_img)
        Dataset=np.load("../Brain_Phantom/Patches_no_air_big/NumpyArchives/NumpyArchiv_" + file_name_queue.decode("utf-8"))
        #Dataset=np.load("../Brain_Phantom/Patches_no_air_big/NumpyArchives/NumpyArchiv_000000.npz")
        Params=Dataset['Params']
        qBOLD=Dataset['qBOLD'] + np.random.normal(loc=0,scale=1./100,size=Dataset['qBOLD'].shape) #max Amplitude = 1, so 1% noise
        QSM=Dataset['QSM'] + np.random.normal(loc=0,scale=0.1/100,size=Dataset['QSM'].shape) #max Amplitude = 0.1, so 1% noise
        #Params.shape
        #qBOLD.shape
        #QSM.shape
        #QSM=QSM[:,:,:,0]
        #qBOLD  = np.array(io.imread('../Brain_Phantom/Patches_no_air_big/qBOLD/qBOLD_' + file_name_queue.decode("utf-8")))
        #qBOLD = np.moveaxis(qBOLD,0,-1)
        #qBOLD = np.expand_dims(qBOLD,axis=0)
        #QSM  = np.array(io.imread('../Brain_Phantom/Patches_no_air_big/QSM/QSM_' + file_name_queue.decode("utf-8")))
        #QSM = np.moveaxis(QSM,0,-1)
        #QSM = np.expand_dims(QSM,axis=[0,-1])
        #Params  = np.array(io.imread('../Brain_Phantom/Patches_no_air_big/Params/Params_' + file_name_queue.decode("utf-8")))
        S0=Params[:,:,:,0]
        S0.shape
        R2=Params[:,:,:,1]
        Y=Params[:,:,:,2]
        nu=Params[:,:,:,3]
        chi_nb=Params[:,:,:,4]

        return qBOLD,QSM,S0,R2,Y,nu,chi_nb
