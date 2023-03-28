import string
import matplotlib.pyplot as plt
import numpy as np
import CNNUtils.imageProcessing as ip
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD


class Model:
    def __init__(self,
                 dataset: any,
                 classes_number: int,
                 input_shape: any,
                 color_scale: int,
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
                 large_filter: int,
                 cmap: string,
                 possibility_precision: int,
                 with_info: bool = True,
                 verbose: int = 1):
        self.__dataset = dataset
        self.__classes_number = classes_number
        self.__input_shape = input_shape
        self.__shape_x, self.__shape_y, self.__shape_z = input_shape
        self.__color_scale = color_scale
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
        self.__cmap = cmap
        self.__possibility_precision = possibility_precision
        self.__with_info = with_info
        self.__verbose = verbose

        self.__log = ""
        self.__is_trained = False
        self.__prediction = None

        # Load the data and split it between train and test sets
        (train_input, train_output), (test_input, test_output) = mnist.load_data()
        # normalizing input data
        train_input = self.__scale_pixels(train_input)
        test_input = self.__scale_pixels(test_input)
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

        self.__model = self.__define_model()

    # ------------------------------------------------------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------------------------------------------------------
    def get_log(self):
        return self.__log

    def train(self):
        try:
            self.__logger("starting load", "train")
            load = keras.models.load_model(self.__save_path)
            if load is None:
                self.__logger("load failed", "train")
                self.__train()
            else:
                self.__logger("load successful", "train")
                self.__model = load
                self.__after_train_processing()
        except:
            self.__train()

    def hard_train(self):
        self.__train()

    def predict_test(self, index: int):
        if not self.__is_trained:
            return
        max_index = len(self.__test_input) - 1
        if index >= max_index or index < 0:
            self.__logger(f"wrong index. Index must be from 0 to {max_index}", "predict_test")
            return
        possibility, value = self.__get_predicted_data(self.__prediction[index])
        self.__show_image(self.__test_input[index],
                          f"predicted test data: {value}, possibility: {possibility:.{self.__possibility_precision}f}")

    def predict_image(self, path: any):
        img_raw = ip.rec_digit(path, self.__shape_x, self.__shape_y, self.__color_scale)
        img4predict = np.array(img_raw).reshape((-1, self.__shape_x, self.__shape_y, self.__shape_z))
        prediction = self.__model.predict(img4predict)
        possibility, value = self.__get_predicted_data(prediction[0])
        self.__show_image(img_raw,
                          f"predicted data: {value}, possibility: {possibility:.{self.__possibility_precision}f}")
        return value

    # ------------------------------------------------------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------------------------------------------------------
    def __logger(self, message: string, prefix: string = ""):
        log = f"[{prefix}] {message}\n"
        if self.__with_info:
            print(log)
        self.__log += log

    def __define_model(self):
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

    def __train(self):
        self.__logger("start train", "__train")
        self.__model.fit(self.__train_input, self.__train_output, epochs=self.__epochs, batch_size=self.__batch_size,
                         verbose=self.__verbose)
        self.__after_train_processing()
        self.__logger("saving", "__train")
        self.__model.save(self.__save_path)
        _, acc = self.__model.evaluate(self.__test_input, self.__test_output, verbose=self.__verbose)
        self.__logger(f"accuracy: {acc}", "__train")

    def __after_train_processing(self):
        self.__logger("start", "__after_train_processing")
        self.__is_trained = True
        self.__prediction = self.__model.predict(self.__test_input)
        self.__logger("end", "__after_train_processing")

    def __show_image(self, image: any, label: string = ""):
        plt.imshow(image, cmap=self.__cmap)
        plt.axis('off')
        plt.title(label)
        plt.show()

    @staticmethod
    def __get_predicted_data(prediction):
        possibility = 0
        result_index = 0
        current_index = 0
        for element in prediction:
            if element > possibility:
                possibility = element
                result_index = current_index
            current_index += 1
        return possibility, result_index

    @staticmethod
    def __scale_pixels(data):
        result = data / data.max()
        return result
