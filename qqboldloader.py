import logging
import os
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from dataloader import Dataloader

# configure logger
logger = logging.getLogger(__name__)

class QQBoldLoader(DataLoader):
    """
    seed : int, optional
        set a fixed seed for the loader, by default 42
    mode : has no effect, should not be Apply
    name : str, optional
        the name of the loader, by default 'reader'
    """

    def __init__(
        self,
        file_dict: Dict[str, Dict[str, str]],
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
    sets all important configurations from the config file:
    - n_channels
    - dtypes
    - dshapes

    also derives:
    - data_rank
    - slice_shift

    """
    # dtypes and dshapes are defined in the base class
    # pylint: disable=attribute-defined-outside-init

    if self.mode is self.MODES.TRAIN or self.mode is self.MODES.VALIDATE:
        self.dtypes = [cfg.dtype, cfg.dtype]
        self.dshapes = [
            np.array(cfg.train_input_shape),
            np.array(cfg.train_label_shape),
        ]
        # use the same shape for image and labels
        assert np.all(
            self.dshapes[0][:2] == self.dshapes[1][:2]
        ), "Sample and label shapes do not match."
    else:
        raise ValueError(f"Not allowed mode {self.mode}")

    self.data_rank = len(self.dshapes[0])

    assert self.data_rank in [3, 4], "The rank should be 3 or 4."

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
    data_img, label_img = self._load_file(file_name_queue)
    samples, labels = self._get_samples_from_volume(data_img, label_img)
    return samples, labels
