import string

import keras.datasets.mnist

# ----------------------------------------------------------------------------------------------------------------------
# Model / data parameters
# ----------------------------------------------------------------------------------------------------------------------
CLASSES_NUMBER = 10
INPUT_SHAPE = 28, 28, 1
DATASET = keras.datasets.mnist.load_data()
COMMON_ACTIVATION = "relu"
LAST_ACTIVATION = "softmax"
KERNEL_INITIALIZER = "he_uniform"

# ----------------------------------------------------------------------------------------------------------------------
# Train parameters
# ----------------------------------------------------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.01
MOMENTUM = 0.9
LOSS = "categorical_crossentropy"
METRICS = ["accuracy"]
SMALL_DOT = (2, 2)
LARGE_DOT = (3, 3)
SMALL_DENSE = 10
LARGE_DENSE = 100
SMALL_FILTER = 32
LARGE_FILTER = 64

# ----------------------------------------------------------------------------------------------------------------------
# Saves
# ----------------------------------------------------------------------------------------------------------------------
SAVE_FILENAME: string = "model_saves/save.h5"
TRAIN_HISTORY_FILENAME = "model_saves/history"