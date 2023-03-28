import string
import cv2


def rec_digit(img_path: string, shape_x: int, shape_y: int, color_scale: int):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gray = color_scale - img

    gray = cv2.resize(gray, (shape_x, shape_y))
    cv2.imwrite('gray' + img_path, gray)
    img = gray / color_scale
    return img
