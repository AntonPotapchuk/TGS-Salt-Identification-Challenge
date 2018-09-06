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