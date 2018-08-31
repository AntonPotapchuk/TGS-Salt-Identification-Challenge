import numpy as np
import pandas as pd
from skimage.transform import resize


def make_submission(test_ids, preds_test, submission_path, original_size=101):
    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                           (original_size, original_size),
                                           mode='constant', preserve_range=True))

    def RLenc(img, order='F', format=True):
        """
        img is binary mask image, shape (r,c)
        order is down-then-right, i.e. Fortran
        format determines if the order needs to be preformatted (according to submission rules) or not

        returns run length as an array or string (if format is True)
        """
        bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
        runs = []  ## list of run lengths
        r = 0  ## the current run length
        pos = 1  ## count starts from 1 per WK
        for c in bytes:
            if (c == 0):
                if r != 0:
                    runs.append((pos, r))
                    pos += r
                    r = 0
                pos += 1
            else:
                r += 1

        # if last run is unsaved (i.e. data ends with 1)
        if r != 0:
            runs.append((pos, r))
            pos += r
            r = 0

        if format:
            z = ''

            for rr in runs:
                z += '{} {} '.format(rr[0], rr[1])
            return z[:-1]
        else:
            return runs

    pred_dict = {fn[:-4]: RLenc(np.round(preds_test_upsampled[i])) for i, fn in enumerate(test_ids)}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(submission_path)
