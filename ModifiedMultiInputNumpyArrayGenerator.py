
import numpy as np
import tensorflow as tf

class ModifiedMultiInputNumpyArrayGenerator(tf.keras.utils.Sequence):
    def __init__(self, npz_files, batch_size, input_keys, label_keys, shuffle=True):
        """
        :param npz_files: List of paths to .npz files.
        :param batch_size: Size of the batch.
        :param input_keys: List of keys for the input data in the .npz file.
        :param label_keys: List of keys for the labels in the .npz file.
        :param shuffle: Whether to shuffle the data at the end of each epoch.
        """
        self.npz_files = npz_files
        self.batch_size = batch_size
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.npz_files))
        self.on_epoch_end()

    def __len__(self):
        # Return the number of batches per epoch
        return int(np.floor(len(self.npz_files) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = [self.npz_files[i] for i in batch_indexes]

        return self.__data_generation(batch_files)

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_files):
        # Generates data containing batch_size samples
        inputs = []
        labels = []

        for file in batch_files:
            data = np.load(file)
            input_data = [data[key] for key in self.input_keys]
            label_data = [data[key] for key in self.label_keys]
            inputs.append(np.concatenate(input_data, axis=-1) if len(input_data) > 1 else input_data[0])
            labels.append(np.concatenate(label_data, axis=-1) if len(label_data) > 1 else label_data[0])

        # Convert lists to numpy arrays
        inputs = np.array(inputs)
        labels = np.array(labels)

        return inputs, labels
