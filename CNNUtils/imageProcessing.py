from scipy.ndimage.measurements import center_of_mass
import math
import cv2
import numpy as np


class ImageProc:

    # ------------------------------------------------------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 transform_matrix: any,
                 shift_scale: int,
                 data_scale: int,
                 shape: any,
                 min_color_sensitivity: int,
                 max_color_sensitivity: int):
        self.__transform_matrix = transform_matrix
        self.__shift_scale = shift_scale
        self.__data_scale = data_scale
        self.__img_shape_x, self.__img_shape_y, self.__img_shape_z = shape
        self.__data_shape_x = self.__get_data_shape(self.__img_shape_x, data_scale)
        self.__data_shape_y = self.__get_data_shape(self.__img_shape_y, data_scale)
        self.__data_shape_z = self.__get_data_shape(self.__img_shape_z, data_scale)
        self.__min_color_sensitivity = min_color_sensitivity
        self.__max_color_sensitivity = max_color_sensitivity

    def __get_full_transform_matrix(self, x, y):
        return [[self.__transform_matrix[0][0], self.__transform_matrix[0][1], x],
                [self.__transform_matrix[1][0], self.__transform_matrix[1][1], y]]

    def __get_best_shift(self, img):
        center_y, center_x = center_of_mass(img)
        shift_x = np.round(self.__img_shape_y / self.__shift_scale - center_x).astype(int)
        shift_y = np.round(self.__img_shape_x / self.__shift_scale - center_y).astype(int)
        return shift_x, shift_y

    def __shift(self, img, shift_x, shift_y):
        transform_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(img, transform_matrix, (self.__img_shape_y, self.__img_shape_x))
        return shifted

    @staticmethod
    def __get_data_shape(data: any, scale: float):
        return int(data * scale)

    # ------------------------------------------------------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------------------------------------------------------

    def transform(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        border = self.__max_color_sensitivity - img
        # border processing
        (thresh, border) = cv2.threshold(border,
                                         self.__min_color_sensitivity,
                                         self.__max_color_sensitivity,
                                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # removing zero rows and cols
        while np.sum(border[0]) == 0:
            border = border[1:]
        while np.sum(border[:, 0]) == 0:
            border = np.delete(border, 0, 1)
        while np.sum(border[-1]) == 0:
            border = border[:-1]
        while np.sum(border[:, -1]) == 0:
            border = np.delete(border, -1, 1)
        rows, cols = border.shape

        # fitting to data size
        if rows > cols:
            factor = self.__data_shape_x / rows
            rows = self.__data_shape_x
            cols = round(cols * factor)
            border = cv2.resize(border, (cols, rows))
        else:
            factor = self.__data_shape_y / cols
            cols = self.__data_shape_y
            rows = round(rows * factor)
            border = cv2.resize(border, (cols, rows))

        # fitting to image size
        cols_padding = (int(math.ceil((self.__img_shape_x - cols) / self.__shift_scale)),
                        int(math.floor((self.__img_shape_x - cols) / self.__shift_scale)))
        rows_padding = (int(math.ceil((self.__img_shape_y - rows) / self.__shift_scale)),
                        int(math.floor((self.__img_shape_y - rows) / self.__shift_scale)))
        border = np.lib.pad(border, (rows_padding, cols_padding))

        # shifting center of mass
        shift_x, shift_y = self.__get_best_shift(border)
        shifted = self.__shift(border, shift_x, shift_y)
        border = shifted

        img = border / self.__max_color_sensitivity
        return img
