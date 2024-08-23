
import numpy as np
import tensorflow as tf

class MultiInputNumpyArrayGenerator(tf.keras.utils.Sequence):
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
        inputs = {key: [] for key in self.input_keys}
        labels = {key: [] for key in self.label_keys}

        for file in batch_files:
            data = np.load(file)
            for key in self.input_keys:
                inputs[key].append(data[key])
            for key in self.label_keys:
                labels[key].append(data[key])

        # Convert lists to numpy arrays
        inputs = {key: np.array(value) for key, value in inputs.items()}
        labels = {key: np.array(value) for key, value in labels.items()}

        # If there's only one input or label, return them directly instead of a dictionary
        if len(inputs) == 1:
            inputs = list(inputs.values())[0]
        if len(labels) == 1:
            labels = list(labels.values())[0]

        return inputs, labels

# Example Usage:

# List of .npz files
#npz_files = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.npz')]

# Parameters
#batch_size = 32
#input_keys = ['images', 'metadata']  # Replace with your actual input keys
#label_keys = ['labels', 'aux_labels']  # Replace with your actual label keys

# Instantiate the generator
#train_generator = MultiInputNumpyArrayGenerator(npz_files, batch_size, input_keys, label_keys)

# Example of using with a model
# model.fit(train_generator, epochs=10)


# Instantiate the generator
#test_generator = MultiInputNumpyArrayGenerator(npz_files, batch_size, input_keys, label_keys)

# Generate a single batch (index 0 for the first batch)
#inputs, labels = test_generator[0]

# Inspect the output shapes
#print("Input Shapes:")
#if isinstance(inputs, dict):
#    for key, value in inputs.items():
#        print(f"{key}: {value.shape}")
#else:
#    print(inputs.shape)

#print("\nLabel Shapes:")
#if isinstance(labels, dict):
#    for key, value in labels.items():
#        print(f"{key}: {value.shape}")
#else:
#    print(labels.shape)