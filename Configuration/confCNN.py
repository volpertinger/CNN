import string

import keras.datasets.mnist

# ----------------------------------------------------------------------------------------------------------------------
# Model / data parameters
# ----------------------------------------------------------------------------------------------------------------------
CLASSES_NUMBER = 10
INPUT_SHAPE = 28, 28, 1
COLOR_SCALE = 255
DATASET = keras.datasets.mnist.load_data()
COMMON_ACTIVATION = "relu"
LAST_ACTIVATION = "softmax"
KERNEL_INITIALIZER = "he_uniform"
CMAP = "binary"

# ----------------------------------------------------------------------------------------------------------------------
# Train parameters
# ----------------------------------------------------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9
VALIDATION_SPLIT = 0.1
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
SAVE_FILENAME: string = "model_saves/save"
TRAIN_HISTORY_FILENAME = "model_saves/history"

# ----------------------------------------------------------------------------------------------------------------------
# Image processing
# ----------------------------------------------------------------------------------------------------------------------
TRANSFORM_MATRIX = [[1, 0], [0, 1]]
SHIFT_SCALE = 2
DATA_SCALE = 0.7
MAX_COLOR_SENSITIVITY = 255
MIN_COLOR_SENSITIVITY = 128

# ----------------------------------------------------------------------------------------------------------------------
# Other
# ----------------------------------------------------------------------------------------------------------------------
WITH_INFO = True
WITH_PLOT = True
VERBOSE = 1
POSSIBILITY_PRECISION = 8
