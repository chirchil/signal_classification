import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import keras

# Set GPU config for tensorflow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

seed = 20
tf.random.set_seed(seed)
np.random.seed(seed)
# Training data path
data_dir = pathlib.Path('Training_Data')
wav_type = np.array(tf.io.gfile.listdir(str(data_dir)))
# Arrays of training file names
filenames_human = tf.io.gfile.glob(str(data_dir) + '/human/*')
filenames_human = tf.random.shuffle(filenames_human)
filenames_spoof = tf.io.gfile.glob(str(data_dir) + '/spoof/*')
filenames_spoof = tf.random.shuffle(filenames_spoof)
# Split training data to train, val and test by 80/10/10
train_files_human = filenames_human[:4000]
val_files_human = filenames_human[4000:4000 + 500]
test_files_human = filenames_human[-500:]
train_files_spoof = filenames_spoof[:4000]
val_files_spoof = filenames_spoof[4000:4000 + 500]
test_files_spoof = filenames_spoof[-500:]
# Concatenate training data
train_files = tf.concat([train_files_human, train_files_spoof], 0)
train_files = tf.random.shuffle(train_files)
test_files = tf.concat([test_files_human, test_files_spoof], 0)
test_files = tf.random.shuffle(test_files)
val_files = tf.concat([val_files_human, val_files_spoof], 0)
val_files = tf.random.shuffle(val_files)
print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))

# .wav file to tensor
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


# Label for each wav (parent directory)
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


# File path to tuple "audio - label"
def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)


# Convert waveform to spectrogram
def get_spectrogram(waveform):
    # Padding for files with less than 200000 samples
    zero_padding = tf.zeros([200000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=256, frame_step=512)
    spectrogram = tf.abs(spectrogram)

    return spectrogram


# Plot spectrogram
def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns). An epsilon is added to avoid log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


# Transform the waveform dataset to "spectrogram-labels"
def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == wav_type)
    return spectrogram, label_id


spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)


# Preprocess dataset for model training
def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds


# Make datasets for train, validation and test
train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)
# Batch the training and validation sets for training
batch_size = 16
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# Get the input shape for model
for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
print('Input shape:', input_shape)
# Labels number
num_labels = len(wav_type)

# AlexNets CNN architecture
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_labels)
])
# Take a look on our model and compile it
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)
# Set epochs to 99
EPOCHS = 99
# Callback to not to stop training if the validation loss will not improve
callback1 = tf.keras.callbacks.EarlyStopping(verbose=1, patience=EPOCHS)
# Callback to save every best model with it's val_loss value (depending on validation loss)
callback2 = tf.keras.callbacks.ModelCheckpoint('model.{epoch:02d}-{val_loss:.4f}.h5',
                                               monitor='val_loss',
                                               verbose=1,
                                               save_best_only=True)
# Fit the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[callback1, callback2],
)
# Plot the model training result
metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

# Make datasets for testing
test_audio = []
test_labels = []
for audio, label in test_ds:
    test_audio.append(audio.numpy())
    test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)
y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels
# Print the test set accuracy
test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')
# Check model performance by creating a confusion matrix
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=wav_type, yticklabels=wav_type,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()
# Testing data path
data_dir_test = pathlib.Path('Testing_Data')
filenames_test = tf.io.gfile.glob(str(data_dir_test) + '/*')
# Create a file with results
f = open('results.txt', 'w')
for i in range(len(filenames_test)):
    # Write results by "sample_number - score"
    # Score is a probability that the sample is human voice
    sample_file = filenames_test[i]
    sample_name = sample_file.split('\\')
    sample_ds = preprocess_dataset([str(sample_file)])
    for spectrogram, label in sample_ds.batch(1):
        # Make a prediction
        prediction = model(spectrogram)
        res = tf.nn.softmax(prediction[0]).numpy()
        f.write(sample_name[1] + ', ' + str(res[0]) + '\n')
