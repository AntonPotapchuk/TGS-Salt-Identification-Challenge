import keras.backend as K
import tensorflow as tf
import numpy as np


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


# Lovasz-Softmax and Jaccard hinge loss in Tensorflow
# Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
# https://github.com/bermanmaxim/LovaszSoftmax/blob/master/tensorflow/lovasz_losses_tf.py
def __lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------
def lovasz_hinge(labels, logits):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
    """
    def treat_image(log_lab):
        log, lab = log_lab
        # log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
        log, lab = __flatten_binary_scores(log, lab)
        return __lovasz_hinge_flat(log, lab)
    losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
    loss = tf.reduce_mean(losses)
    loss.set_shape((None,))
    return loss


def __lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = __lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    #loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
    #               lambda: tf.reduce_sum(logits) * 0.,
    #               compute_loss,
    #               strict=True,
    #               name="loss"
    #               )
    loss = compute_loss()
    return loss


def __flatten_binary_scores(scores, labels):
    """
    Flattens predictions in the batch (binary case)
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    return scores, labels


# My loss
def __mean_prec_iou(y_true, y_pred):
    with tf.variable_scope(None, 'mean_prec_iou', (y_pred, y_true)):
        y_pred.get_shape().assert_is_compatible_with(y_true.get_shape())
        y_pred = tf.cast(tf.round(y_pred), tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        threasholds_iou = tf.constant(np.arange(0.5, 1.0, 0.05), dtype=tf.float32)

        def intersection_over_union(masks):
            y_true, y_pred = masks
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])
            intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
            # intersection = tf.Print(intersection, [intersection], message='Intersection:')
            union = tf.reduce_sum(tf.add(y_true, y_pred)) - intersection
            # union = tf.Print(union, [union], message='Union:')
            iou = tf.where(tf.equal(union, 0), tf.constant(0., dtype=tf.float32),
                           tf.cast(intersection / union, tf.float32))
            # iou = tf.Print(iou, [iou], message='IOU:')
            greater = tf.cast(tf.greater(iou, threasholds_iou), tf.float32)
            score = tf.reduce_mean(greater)
            # score = tf.Print(score, [score], message='Score')
            score = tf.where(tf.logical_and(tf.equal(tf.round(tf.reduce_sum(y_true)), 0),
                                            tf.equal(tf.round(tf.reduce_sum(y_pred)), 0)),
                             tf.constant(1., dtype=tf.float32), score)
            return score

        batch_score = tf.map_fn(intersection_over_union, (y_true, y_pred), dtype=tf.float32)
        # batch_score = tf.Print(batch_score, [batch_score], 'Batch score:')
        batch_score = tf.reduce_mean(batch_score)
        # batch_score = tf.Print(batch_score, [batch_score])
        return batch_score

def mean_iou_loss(y_true, y_pred):
    return K.constant(1., dtype=tf.float32) - __mean_prec_iou(y_true, y_pred)