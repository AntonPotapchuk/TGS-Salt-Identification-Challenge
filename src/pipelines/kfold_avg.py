import gc
import numpy as np
import os
import sys

from argparse import ArgumentParser
from data.loading import get_dataset
from data.datagen import create_datagen
from data.submission import make_submission
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from models import get_model, get_model_class, reset_tensorflow
from models.losses import lovasz_loss
from models.metrics import my_iou_metric, my_iou_metric_2
from tqdm import tqdm
from sklearn.model_selection import KFold


def get_callbacks(model_path, args, tensorboard_dir=None):
    callbacks = []
    if args.early_stopping is not None and args.early_stopping > 0:
        monitor = "loss" if args.save_not_best_only else "val_loss"
        callbacks.append(EarlyStopping(patience=args.early_stopping, verbose=1, monitor=monitor))
    if tensorboard_dir is not None:
        callbacks.append(TensorBoard(tensorboard_dir, write_graph=False))
    if model_path is not None:
        monitor = "loss" if args.save_not_best_only else "val_loss"
        callbacks.append(ModelCheckpoint(model_path, save_best_only=True, monitor=monitor))
    if args.reduce_lr_patience is not None and args.reduce_lr_patience > 0:
        callbacks.append(ReduceLROnPlateau(monitor='loss',
                                           factor=args.reduce_lr_alpha,
                                           patience=args.reduce_lr_patience))
    return callbacks


def create_parser():
    from data.datagen import add_argparser_arguments as add_datagen_args
    parser = ArgumentParser(description="Single model pipeline")
    parser.add_argument('--name', help='Experiment name (experiment)', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-name', default='baseline')
    parser.add_argument('--log-path', default='./')
    parser.add_argument('--random-seed', default=123, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--shuffle', default=True, action='store_true')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--dropout', help='Fraction of the input units to drop', default=0, type=float)
    parser.add_argument('--early-stopping', help='early-stopping-patience', default=9, type=int)
    parser.add_argument('--reduce-lr-patience', default=3, type=int)
    parser.add_argument('--reduce-lr-alpha', default=0.2, type=float)
    parser.add_argument('--save-not-best-only', default=False, action='store_true')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--nfolds', default=5, type=int)
    parser = add_datagen_args(parser)
    return parser


def training_stage(train_gen, val_gen, train_steps, val_steps, model_path, tensorboard_dir, args, last_activation,
                metrics, loss, X_train, Y_train, X_val, Y_val, weights_path=None):
    print("Creating model")
    sys.stdout.flush()
    reset_tensorflow()
    model = get_model(args.model_name, dropout=args.dropout, last_activation=last_activation)
    model.compile(optimizer=args.optimizer, loss=loss, metrics=metrics)
    if weights_path is not None:
        print("Loading weights")
        model.load_weights(weights_path)
    callbacks = get_callbacks(model_path, args, tensorboard_dir=tensorboard_dir)
    print("Fitting model")
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=train_steps,
                        epochs=args.epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        shuffle=args.shuffle)
    model.load_weights(model_path)
    print("Evaluating")
    res = model.evaluate(X_train, Y_train, verbose=0)
    print("Train loss:", res[0])
    print("Train mean IoU:", res[1])
    res = model.evaluate(X_val, Y_val, verbose=0)
    print("Test loss:", res[0])
    print("Test mean IoU:", res[1])
    return model


def predict_result(model, x_test):  # predict both orginal and reflect x
    x_test_reflect = np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test)
    preds_test2_refect = model.predict(x_test_reflect)
    preds_test += np.array([np.fliplr(x) for x in preds_test2_refect])
    return preds_test / 2.


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


def pipeline(args):
    ############################# INITIALIZATION #############################
    args.validation_split = 0
    batch_size = args.batch_size
    shuffle = args.shuffle
    random_seed = args.random_seed

    experiment_name = args.name
    data_path = args.data_path
    log_path = args.log_path

    tensorboard_dir_base = os.path.join(log_path, 'tensorboard', experiment_name)
    submission_dir = os.path.join(log_path, 'submission')
    submission_path_template = os.path.join(submission_dir, experiment_name + '%s.csv')
    model_dir = os.path.join(log_path, './checkpoints')
    model_path_template = os.path.join(model_dir, experiment_name + '_fold%d.h5')

    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    train_img_path = os.path.join(train_path, 'images')
    train_mask_path = os.path.join(train_path, 'masks')
    test_img_path = os.path.join(test_path, 'images')

    if not os.path.isdir(submission_dir):
        os.makedirs(submission_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    ############################# GET IMAGE PREPROCESSING INFO #############################
    model_class = get_model_class(args.model_name)
    image_process_func = model_class.get_image_preprocessor()
    image_size = model_class.get_image_size()
    n_channels = model_class.get_number_of_channels()

    ############################# CREATE TRAIING SET #############################
    images_ids = os.listdir(train_img_path)
    print("Preparing training set")
    sys.stdout.flush()
    images, masks = get_dataset(images_ids, train_img_path, train_mask_path, image_size=image_size, is_test=False,
                                preprocess_func=image_process_func, single_channel=n_channels==1)
    print("Preparing test set")
    sys.stdout.flush()
    test_ids = os.listdir(test_img_path)
    X_test = get_dataset(test_ids, test_img_path, None, image_size=image_size, is_test=True,
                         preprocess_func=image_process_func, single_channel=n_channels==1)
    print("Train shape:", images.shape)
    print("Test shape:", X_test.shape)
    sys.stdout.flush()

    ############################# K-FOLD TRAINING #############################
    kf = KFold(n_splits=args.nfolds, random_state=args.random_seed, shuffle=True)
    predictions = np.zeros(shape=(args.nfolds, len(test_ids), image_size, image_size, 1), dtype=np.float32)
    for fold, (train_index, test_index) in enumerate(kf.split(images, masks)):
        print("\n\nFold: ", fold + 1)
        tensorboard_dir = tensorboard_dir_base + "_fold" + str(fold + 1)
        model_path = model_path_template % (fold + 1)
        submission_path = submission_path_template % ("_fold" + str(fold + 1))
        if not os.path.isdir(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        gc.collect()
        # Prepairing data
        X_train, X_val = images[train_index], images[test_index]
        Y_train, Y_val = masks[train_index], masks[test_index]
        train_gen = create_datagen(X_train, Y_train, args,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   random_seed=random_seed)
        val_gen = create_datagen(X_val, Y_val, args,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 random_seed=random_seed)
        train_steps = int(len(X_train) / batch_size)
        val_steps = int(len(X_val) / batch_size)
        gc.collect()

        ############################# Creating model #############################
        print("Stage 1: binary crossentropy loss")
        training_stage(train_gen, val_gen, train_steps, val_steps, model_path,
                       tensorboard_dir, args, "sigmoid", [my_iou_metric], "binary_crossentropy",
                       X_train, Y_train, X_val, Y_val)
        print("Stage 2: lovasz loss")
        model = training_stage(train_gen, val_gen, train_steps, val_steps, model_path,
                       tensorboard_dir, args, "linear", [my_iou_metric_2], lovasz_loss,
                       X_train, Y_train, X_val, Y_val, model_path)
        model.load_weights(model_path)

        print("Validation prediction. Estimating optimal threshold.")
        preds_valid = predict_result(model, X_val)
        # Scoring for last model, choose threshold by validation data
        thresholds_ori = np.linspace(0.3, 0.7, 31)
        # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
        thresholds = np.log(thresholds_ori / (1 - thresholds_ori))
        ious = np.array([iou_metric_batch(Y_val, preds_valid > threshold) for threshold in tqdm(thresholds)])
        print("IOUS:", ious)
        # instead of using default 0 as threshold, use validation data to find the best threshold.
        threshold_best_index = np.argmax(ious)
        iou_best = ious[threshold_best_index]
        threshold_best = thresholds[threshold_best_index]
        print("Best threshold: %f. Best iou: %f." % (threshold_best, iou_best))
        print("Test prediction")
        preds_test = predict_result(model, X_test)
        predictions[fold] = preds_test
        print("Making submission")
        make_submission(test_ids, preds_test, submission_path, mode="threshold", threshold=threshold_best)

    predictions = np.mean(predictions, axis=0)
    np.save(os.path.join(submission_dir, args.model_name + "_prediction.npy"), predictions)
    print("\n\nMaking final submission")
    # TODO ???? > 0 ????
    make_submission(test_ids, predictions, submission_path_template % "_final", mode="threshold", threshold=0.0)


if __name__ == "__main__":
    pipeline(create_parser().parse_args())
