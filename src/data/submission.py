import numpy as np
import pandas as pd
from skimage.transform import resize
from common.constants import ORIGINAL_IMAGE_SIZE, MAX_WIDTH_WITHOUT_PADDING
from data.common import get_padding_width


def transform_predictions(prediction, original_size):
    prediction = np.squeeze(prediction)
    size = prediction.shape[0]
    pad_width = get_padding_width(size, MAX_WIDTH_WITHOUT_PADDING)
    prediction = prediction[pad_width:-pad_width, pad_width:-pad_width]
    prediction = resize(prediction, (original_size, original_size), mode='constant', preserve_range=True)
    return prediction


def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def make_submission(test_ids, preds_test, submission_path):
    # Create list of upsampled test masks
    predictions = []
    for i in range(len(preds_test)):
        predictions.append(transform_predictions(preds_test[i], ORIGINAL_IMAGE_SIZE))

    pred_dict = { fn[:-4]: rle_encode(np.round(predictions[i])) for i, fn in enumerate(test_ids)}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(submission_path)
