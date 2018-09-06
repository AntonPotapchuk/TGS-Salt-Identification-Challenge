import os
import numpy as np

from common.constants import MAX_WIDTH_WITHOUT_PADDING
from data.common import get_padding_width
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from matplotlib import pyplot as plt

def process_image(img, width, preprocess_func=None):
    if width <= MAX_WIDTH_WITHOUT_PADDING:
        img = resize(img, (width, width), mode='constant', preserve_range=True)
    else:
        img = resize(img, (MAX_WIDTH_WITHOUT_PADDING, MAX_WIDTH_WITHOUT_PADDING), mode='constant', preserve_range=True)
        pad_width = get_padding_width(width, MAX_WIDTH_WITHOUT_PADDING)
        pad_width = [(pad_width, pad_width), (pad_width, pad_width)]
        if len(img.shape) == 3:
            pad_width.append((0, 0))
        img = np.pad(img, mode='reflect', pad_width=pad_width)
    if preprocess_func is not None:
        img = preprocess_func(img)
    return img


def get_dataset(ids, img_folder, mask_folder, image_size, is_test=False, preprocess_func=None):
    X = []
    if not is_test:
        Y = np.zeros((len(ids), image_size, image_size, 1), dtype=np.bool)
    for n, id_ in enumerate(ids):
        img = load_img(os.path.join(img_folder, id_))
        img = img_to_array(img)
        img = process_image(img, image_size, preprocess_func)
        X.append(img)
        if not is_test:
            mask = load_img(os.path.join(mask_folder, id_))
            mask = img_to_array(mask)[:,:,1]
            Y[n] = np.expand_dims(process_image(mask, image_size, None), axis=2)

    X = np.nan_to_num(np.array(X))
    if is_test:
        return X
    return X, Y
