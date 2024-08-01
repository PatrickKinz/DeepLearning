"""!
Contains the data loader class, which should be subclassed.
"""

import logging
from abc import abstractmethod
from collections.abc import Collection
from enum import Enum
from typing import Any, List, Tuple

import numpy as np
import tensorflow as tf

# configure logger
logger = logging.getLogger(__name__)


class DataLoader(object):
    """This is the Basic Class for reading data using the Tensorflow Dataset Pipeline.
    Inheriting Classes need to implement:
    - _read_file_and_return_numpy_samples()

    Parameters
    ----------
    mode : MODES, optional
        The mode to use, if None, train will be used, by default None
    seed : int, optional
        The seed to use for random sampling, by default 42
    name : str, optional
        The name of the dataloader, by default 'reader'
    n_inputs : int, optional
        The number of elements for x, by default 1
    n_labels : int, optional
        The number of elements per label, if None, it is set to 1 during training
        and otherwise to 0, by default None
    shuffle : bool, optional
        If the dataset should be shuffled, by default None
        If none, dataset will be shuffled for modes TRAIN and VALIDATE
    sample_buffer_size : int, optional
        The size of the buffer that will be shuffled, by default None
        If it is none, 8 times the number of files will be used.
    """

    class MODES(Enum):
        """!
        Possible Modes for Reader Classes:
        - TRAIN = 'train': from list of file names, data augmentation and shuffling, drop remainder
        - VALIDATE = 'validate': from list of file names, shuffling, drop remainder
        - APPLY = 'apply' from list single file id, in correct order, remainder in last smaller batch
        """

        TRAIN = "train"
        VALIDATE = "validate"
        APPLY = "apply"

    def __init__(
        self,
        mode=None,
        seed=42,
        name="reader",
        n_inputs=1,
        n_labels=None,
        shuffle=None,
        sample_buffer_size=None,
    ):

        if mode is None:
            self.mode = self.MODES.TRAIN
        else:
            self.mode = mode

        if shuffle is None:
            if mode == self.MODES.APPLY:
                self.shuffle = False
            else:
                self.shuffle = True
        else:
            self.shuffle = shuffle

        if mode == self.MODES.APPLY and self.shuffle:
            raise ValueError("For applying, shuffle should be turned off.")

        if self.mode not in self.MODES:
            raise ValueError("mode = '{}' is not supported by network".format(mode))

        # set the number of elements in x
        self.n_inputs = n_inputs
        # determine the number of label elements if not set
        if n_labels is None:
            if self.mode == self.MODES.TRAIN:
                self.n_labels = 1
            else:
                self.n_labels = 0
        else:
            self.n_labels = n_labels

        self.name = name
        self.seed = seed
        self.sample_buffer_size = sample_buffer_size
        self.dshapes = None
        self.dtypes = None
        self.n_files = None

        # set seed
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        self._set_up_shapes_and_types()

    @abstractmethod
    def _set_up_shapes_and_types(self):
        """This function should set the following properties:
        - dshapes for the data shape for the returned elements for example (sample_shape, label_shape)
        - dtypes should contain the corresponding datatypes in the tuple
        """
        raise NotImplementedError("not implemented")

    def __call__(
        self, file_list: List, batch_size: int, n_epochs=50, read_threads=1
    ) -> tf.data.Dataset:
        """This function operates as follows,
        - Generates Tensor of strings from the file_list
        - Creates file_list_ds dataset from the Tensor of strings.
        - If loader is in training mode (self.mode == 'train'),
            - file_list_ds is repeated with shuffle n_epoch times
            - file_list_ds is mapped with _read_wrapper() to obtain the dataset.
            The mapping generates a set of samples from each pair of data and label files identified by each file ID.
            Here each element of dataset is a set of samples corresponding to one ID.
            - dataset is flat mapped to make each element correspond to one sample inside the dataset.
            - dataset is shuffled
            - dataset is batched with batch_size
            - 1 element of dataset is prefetched
            - datset is returned

        - Else if loader is in validation mode (self.mode == 'validation'),
            - file_list_ds is mapped with _read_wrapper() to obtain dataset (mapping is same as train mode)
            - dataset is flat mapped to make each element correspond to one sample inside the dataset.
            - dataset is batched with batch_size
            - dataset is returned

        Parameters
        ----------
        file_list : List
            array of strings, where each string is a file ID corresponding to a pair of
            data file and label file to be loaded. file_list should be obtained from a .csv file
            and then converted to numpy array. Each ID string should have the format 'Location\\file_number'.
            From Location, the data file and label file with the file_number, respectively
            named as volume-file_number.nii and segmentation-file_number.nii are loaded.
            (See also LitsLoader._read_file(), LitsLoader._load_file() for more details.)
        batch_size : int
            The batch size
        n_epochs : int, optional
            The number of training epochs, by default 50
        read_threads : int, optional
            int, number of threads/instances to read in parallel, by default 1

        Returns
        -------
        tf.data.Dataset
            tf.dataset of data and labels
        """

        if not np.issubdtype(type(batch_size), int):
            raise ValueError("The batch size should be an integer")

        if self.mode is self.MODES.APPLY:
            self.n_files = 1
        else:
            self.n_files = len(file_list)

        # set the buffer size
        if self.sample_buffer_size is None:
            sample_buffer_size = 8 * self.n_files
        else:
            sample_buffer_size = self.sample_buffer_size

        with tf.name_scope(self.name):
            id_tensor = tf.convert_to_tensor(file_list, dtype=tf.string)

            # Create dataset from list of file names
            if self.mode is self.MODES.APPLY:
                file_list_ds = tf.data.Dataset.from_tensors(id_tensor)
            else:
                file_list_ds = tf.data.Dataset.from_tensor_slices(id_tensor)

            if self.mode is self.MODES.TRAIN:
                # shuffle and repeat n_epoch times if in training mode
                file_list_ds = file_list_ds.shuffle(buffer_size=self.n_files)#.repeat(                  count=n_epochs                )

            # read data from file using the _read_wrapper
            dataset = file_list_ds.map(
                map_func=self._read_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE #was num_parallel_calls=read_threads
            )
            # in the result, each element in the dataset has the number of samples
            # per file as first dimension followed by the sample shape

            # this function will flatten the datasets, so that each element has
            # the shape of the sample
            dataset = self._zip_data_elements_tensorwise(dataset)

            # zip the datasets, so that it is in the format (x, y)
            if self.n_inputs != 1 or self.n_labels > 1:
                dataset = dataset.map(self._make_x_y)

            if self.mode is not self.MODES.APPLY:
                # shuffle
                if self.shuffle:
                    dataset = dataset.shuffle(
                        buffer_size=sample_buffer_size, seed=self.seed
                    )

            if self.mode is self.MODES.APPLY:
                dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
            else:
                # no smaller final batch
                dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

            # batch prefetch
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) #Patrick: was originally 1

        return dataset

    def _zip_data_elements_tensorwise(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """here each element corresponds to one file.
        Use flat map to make each element correspond to one Sample.
        If there is more than one element, they will be zipped together. This could
        be (sample, label) or more elements.

        Parameters
        ----------
        ds : tf.data.Dataset
            Dataset

        Returns
        -------
        ds : tf.data.Dataset
            Dataset where each element corresponds to one sample

        """
        if len(dataset.element_spec) == 1:
            dataset = dataset.flat_map(lambda e: tf.data.Dataset.from_tensor_slices(e))
        else:
            # interleave the datasets, so that the order is the first sample from
            # all images, then the second one and so on. This results in better
            # shuffling, because not all sample from one image follow each other.
            dataset = dataset.interleave(
                lambda *elem: tf.data.Dataset.zip(
                    tuple((tf.data.Dataset.from_tensor_slices(e) for e in elem))
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        return dataset

    def _make_x_y(self, *datasets) -> List[Any]:
        """This function takes a variable number of arguments and turns them into
        a list with 2 elements for x and y depending on the specified number of
        inputs and labels

        Returns
        -------
        List[Any]
            The list
        """
        output = []
        # if there is more than 1 input, turn it into a tuple
        if self.n_inputs > 1:
            output.append(tuple((ds for ds in datasets[: self.n_inputs])))
        else:
            output.append(datasets[0])

        # if there is more than 1 label, turn it into a tuple
        if self.n_labels == 1:
            output.append(datasets[self.n_inputs])
        elif self.n_labels > 1:
            output.append(
                tuple(
                    (ds for ds in datasets[self.n_inputs : self.n_inputs + self.n_labels])
                )
            )
        return output

    def _read_wrapper(self, id_data_set: List[tf.Tensor]) -> List[tf.Tensor]:
        """Wrapper for the _read_file() function
        Wraps the _read_file() function and handles tensor shapes and data types
        this has been adapted from https://github.com/DLTK/DLTK

        Parameters
        ----------
        id_data_set : list
            list of tf.Tensors from the id_list queue. Provides an identifier for the examples to read.
        kwargs :
            additional arguments for the '_read_sample function'

        Returns
        -------
        list
            list of tf.Tensors read for this example
        """

        def get_sample_tensors_from_file_name(file_id):
            """Wrapper for the python function
            Handles the data types of the py_func

            Parameters
            ----------
            file_id : list
            list of tf.Tensors from the id_list queue. Provides an identifier for the examples to read.

            Returns
            -------
            list
                list of things just read


            """
            try:
                samples_np = self._read_file_and_return_numpy_samples(file_id.numpy())
            except Exception as exc:
                logger.exception("got error %s from _read_file: %s", exc, file_id)
                raise
            return samples_np

        ex = tf.py_function(
            get_sample_tensors_from_file_name, [id_data_set], self.dtypes
        )  # use python function in tensorflow

        tensors = []
        # set shape of tensors for downstream inference of shapes
        for tensor_shape, sample_shape in zip(ex, self.dshapes):
            if isinstance(sample_shape, Collection):
                shape: List[Any] = [None] + list(sample_shape)
            else:
                assert sample_shape == 1, "If shape is not 1, use an iterable."
                shape = [None, 1]
            tensor_shape.set_shape(shape)
            tensors.append(tensor_shape)

        return tensors

    @abstractmethod
    def _read_file_and_return_numpy_samples(
        self, file_name_queue: bytes
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("not implemented")
