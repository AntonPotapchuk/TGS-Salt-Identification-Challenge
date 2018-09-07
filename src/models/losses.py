import keras.backend as K


def __dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    smooth = K.constant(smooth)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (K.constant(2.) * intersection + smooth) / (K.sum(y_true, -1) + K.sum(y_pred, -1) + smooth)


def dice_loss(y_true, y_pred):
    return K.constant(1) - __dice_coef(y_true, y_pred)


def jaccard_distance_loss(y_true, y_pred, smooth=1e-6):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    smooth = K.constant(smooth)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.constant(1) - jac
