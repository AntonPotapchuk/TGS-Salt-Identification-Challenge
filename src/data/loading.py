import os
import numpy as np

from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize


def add_argparser_arguments(parser):
    parser.add_argument('--use-first-der', default=False, action='store_true')
    parser.add_argument('--use-second-der', default=False, action='store_true')
    parser.add_argument('--use-sharp', default=False, action='store_true')
    parser.add_argument('--use-canny', default=False, action='store_true')
    parser.add_argument('--use-hog', default=False, action='store_true')
    parser.add_argument('--use-other-features', default=False, action='store_true')
    parser.add_argument('--image-size', default=128, type=int)
    return parser

def resize_img(img, size):
    max_resize = 156
    if size <= max_resize:
        return resize(img, (size, size), mode='constant', preserve_range=True)
    img = resize(img, (max_resize, max_resize), mode='constant', preserve_range=True)
    img = np.pad(img, int((size-max_resize) / 2), mode='constant')
    return img

# TODO: Optimize??
def get_dataset(ids, img_folder, mask_folder, args, is_test=False):
    from data.feature_engineering import get_first_derivatives, get_second_derivatives, \
        get_sharpened_img, get_canny_features, get_hog_features, get_other_features

    im_size = args.image_size
    X = []
    if not is_test:
        Y = np.zeros((len(ids), im_size, im_size, 1), dtype=np.bool)
    for n, id_ in enumerate(ids):
        features = []
        img = load_img(os.path.join(img_folder, id_))
        x = img_to_array(img)[:, :, 0]
        x = resize_img(x, im_size)
        x /= 255
        features.append(x)
        if args.use_first_der:
            first_der = get_first_derivatives(x)
            features.extend(list(first_der))
        if args.use_second_der:
            second_der = get_second_derivatives(x)
            features.extend(list(second_der))
        if args.use_sharp:
            features.append(get_sharpened_img(x))
        if args.use_canny:
            features.append(get_canny_features(x))
        if args.use_hog:
            features.append(get_hog_features(x))
        if args.use_other_features:
            features.extend(list(get_other_features(x)))
        # TODO: Probably, not the best solution
        features = np.moveaxis(np.array(features), 0, 2)
        X.append(features)
        if not is_test:
            mask = img_to_array(load_img(os.path.join(mask_folder, id_)))[:, :, 1]
            # Probably we don't need second axis
            Y[n] = np.expand_dims(resize_img(mask, im_size), axis=2)
    X = np.nan_to_num(np.array(X))
    if is_test:
        return X
    return X, Y