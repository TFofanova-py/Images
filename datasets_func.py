import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random


def get_x_i(image):
    return np.array(image).ravel()


def get_data(dir_name):
    images_list = os.listdir(dir_name)
    random.shuffle(images_list)
    x_data = None
    y_data = np.array([])

    for img_source in images_list:
        img = cv2.imread(f'{dir_name}/{img_source}', cv2.IMREAD_REDUCED_GRAYSCALE_8)
        x_i = get_x_i(img)
        x_data = np.append(x_data, [np.array(x_i)], axis=0) if x_data is not None else np.array([np.array(x_i)])

        y_i = int(img_source.split('__')[0][3:])
        y_data = np.append(y_data, y_i)
    return x_data, y_data


def split_dataset(x_data, y_data):
    n = int(x_data.shape[0])
    n_train = round(n * 0.7)
    return x_data[:n_train, :], y_data[:n_train], x_data[n_train:, :], y_data[n_train:]


def visualization(data, labels):
    n = len(data)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(4, 4))
    for a, im, l in zip(axes, data, labels):
        a.set_axis_off()
        a.imshow(im)
        a.set_title(l)
    plt.show()
