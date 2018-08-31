import numpy as np

from scipy import ndimage, signal
from skimage import exposure
from skimage.feature import canny, hog


def get_first_derivatives(img):
    # http://www.mif.vu.lt/atpazinimas/dip/FIP/fip-Derivati.html
    # https://www.kaggle.com/meaninglesslives/data-exploration
    # Sobel filter: https://en.wikipedia.org/wiki/Image_derivatives
    img_x = ndimage.sobel(img, axis=-1)
    img_y = ndimage.sobel(img, axis=0)
    grad_img = np.hypot(img_x, img_y)
    # Not sure.
    # https://ru.wikipedia.org/wiki/%D0%9E%D0%BF%D0%B5%D1%80%D0%B0%D1%82%D0%BE%D1%80_%D0%A1%D0%BE%D0%B1%D0%B5%D0%BB%D1%8F
    direction = np.arctan(img_y / img_x)
    return img_x, img_y, grad_img, direction


# TODO!!! NOT SURE IT IS CORRECT!!!!!
def get_second_derivatives(img):
    # https://dsp.stackexchange.com/questions/10605/kernels-to-compute-second-order-derivative-of-digital-image
    xder2 = np.array([[1, -2, 1], [2, -4, 2], [1, -2, 1]])
    yder2 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
    img_x = signal.convolve2d(img, xder2, mode='same')
    img_y = signal.convolve2d(img, yder2, mode='same')
    grad_img = np.hypot(img_x, img_y)
    return img_x, img_y, grad_img


# https://www.scipy-lectures.org/advanced/image_processing/auto_examples/plot_sharpen.html
# Need to play with parameters
def get_sharpened_img(img, alpha=30, sigma1=3, sigma2=1):
    img = img.astype(float)
    blurred_f = ndimage.gaussian_filter(img, sigma1)
    filter_blurred_f = ndimage.gaussian_filter(blurred_f, sigma2)
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    return sharpened


def get_canny_features(img):
    return canny(img).astype(np.uint8)


# I don't know about all stuff beneath
# https://www.kaggle.com/meaninglesslives/simple-features
def get_hog_features(img):
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
    return hog_image


# Too lazy to separate them
# https://www.kaggle.com/meaninglesslives/simple-features
def get_other_features(img):
    img = img.astype(np.int8)
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    # Equalization
    img_eq = exposure.equalize_hist(img)
    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img)
    return img_rescale, img_eq, img_adapteq