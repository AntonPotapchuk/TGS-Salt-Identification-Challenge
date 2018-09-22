import tensorflow as tf
import numpy as np


def mean_prec_iou(y_true, y_pred):
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

        batch_score = tf.map_fn(intersection_over_union, (y_true, y_pred), back_prop=False, dtype=tf.float32)
        # batch_score = tf.Print(batch_score, [batch_score], 'Batch score:')
        batch_score = tf.reduce_mean(batch_score)
        # batch_score = tf.Print(batch_score, [batch_score])
        return batch_score


def get_iou_vector(labels, preds):
    batch_size = labels.shape[0]
    metric = []
    for batch in range(batch_size):
        true, predicted = labels[batch] > 0, preds[batch] > 0
        intersection = np.logical_and(true, predicted)
        union = np.logical_or(true, predicted)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))
    return np.mean(metric)


def my_iou_metric(labels, preds):
    return tf.py_func(get_iou_vector, [labels, preds > 0.5], tf.float64)


def my_iou_metric_2(labels, preds):
    return tf.py_func(get_iou_vector, [labels, preds > 0], tf.float64)


# Score the model and do a threshold optimization by the best IoU.
# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(labels, y_pred, print_table=False):
    # if all zeros, original code  generate wrong  bins [-0.5 0 0.5],
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0, 0.5, 1], [0, 0.5, 1]))
    #     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
    # print(temp1)
    intersection = temp1[0]
    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=[0, 0.5, 1])[0]
    area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    # Compute union
    union = area_true + area_pred - intersection
    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    intersection[intersection == 0] = 1e-9
    union = union[1:, 1:]
    union[union == 0] = 1e-9
    # Compute the intersection over union
    iou = intersection / union
    # Precision helper function

    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn
    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)