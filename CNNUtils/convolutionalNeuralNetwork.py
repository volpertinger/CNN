import string

import numpy as np
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# todo: save\load logs
# todo: проверка на сохранение перед train
class Model:
    def __init__(self,
                 dataset: any,
                 classes_number: int,
                 input_shape: any,
                 batch_size: int,
                 epochs: int,
                 save_path: string,
                 history_path: string,
                 common_activation: string,
                 last_activation: string,
                 kernel_initializer: string,
                 learning_rate: float,
                 momentum: float,
                 loss: string,
                 metrics: list,
                 small_dot: any,
                 large_dot: any,
                 small_dense: int,
                 large_dense: int,
                 small_filter: int,
                 large_filter: int):
        self.__dataset = dataset
        self.__classes_number = classes_number
        self.__input_shape = input_shape
        self.__shape_x, self.__shape_y, self.__shape_z = input_shape
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__save_path: string = save_path
        self.__history_path = history_path
        self.__common_activation = common_activation
        self.__last_activation = last_activation
        self.__kernel_initializer = kernel_initializer
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__loss = loss
        self.__metrics = metrics
        self.__small_dot = small_dot
        self.__large_dot = large_dot
        self.__small_dense = small_dense
        self.__large_dense = large_dense
        self.__small_filter = small_filter
        self.__large_filter = large_filter

        self.__log = ""
        self.__is_trained = False

        # Load the data and split it between train and test sets
        (train_input, train_output), (test_input, test_output) = mnist.load_data()
        # reshape dataset to have a single channel
        self.__train_input = train_input.reshape((train_input.shape[0], self.__shape_x, self.__shape_y, self.__shape_z))
        self.__test_input = test_input.reshape((test_input.shape[0], self.__shape_x, self.__shape_y, self.__shape_z))
        # one hot encode target values
        self.__train_output = to_categorical(train_output)
        self.__test_output = to_categorical(test_output)
        # logs
        self.__logger(f"data shape: {self.__input_shape}")
        self.__logger(f"train samples number: {self.__train_input.shape[0]}")
        self.__logger(f"test samples number: {self.__test_input.shape[0]}")

        self.__model = self.define_model()

    # ------------------------------------------------------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------------------------------------------------------
    def get_log(self):
        return self.__log

    def train(self):
        try:
            load = self.__model.load_weights(self.__save_path)
            if load is None:
                self.__train()
            else:
                self.__model = load
                self.__is_trained = True
        except:
            self.__train()

    # ------------------------------------------------------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------------------------------------------------------
    def __logger(self, message: string, prefix: string = ""):
        self.__log += f"[{prefix}] {message}\n"

    def define_model(self):
        model = Sequential()
        model.add(
            Conv2D(self.__small_filter, self.__large_dot, activation=self.__common_activation,
                   kernel_initializer=self.__kernel_initializer,
                   input_shape=self.__input_shape))
        model.add(MaxPooling2D(self.__small_dot))
        model.add(Conv2D(self.__large_filter, self.__large_dot, activation=self.__common_activation,
                         kernel_initializer=self.__kernel_initializer))
        model.add(Conv2D(self.__large_filter, self.__large_dot, activation=self.__common_activation,
                         kernel_initializer=self.__kernel_initializer))
        model.add(MaxPooling2D(self.__small_dot))
        model.add(Flatten())
        model.add(Dense(self.__large_dense, activation=self.__common_activation,
                        kernel_initializer=self.__kernel_initializer))
        model.add(Dense(self.__small_dense, activation=self.__last_activation))
        opt = SGD(learning_rate=self.__learning_rate, momentum=self.__momentum)
        model.compile(optimizer=opt, loss=self.__loss, metrics=self.__loss)
        return model

    # run the test harness for evaluating a model
    def __train(self):
        self.__model.fit(self.__train_input, self.__train_output, epochs=self.__epochs, batch_size=self.__batch_size,
                         verbose=1)
        self.__is_trained = True
        self.__model.save(self.__save_path)
        _, acc = self.__model.evaluate(self.__test_input, self.__test_output, verbose=0)
        self.__logger('> %.3f' % (acc * 100.0))

    @staticmethod
    def __scale_pixels(train, test):
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        # todo: константы
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        return train_norm, test_norm
