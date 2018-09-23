import gc
import os
import sys
import numpy as np

from argparse import ArgumentParser
from data.loading import get_dataset
from data.datagen import create_datagen
from data.submission import make_submission
from models import get_model, get_model_class, get_callbacks, reset_tensorflow
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from models.losses import lovasz_loss
from models.metrics import my_iou_metric, my_iou_metric_2, iou_metric_batch
from tqdm import tqdm


def training_stage(train_gen, val_gen, train_steps, val_steps, model_path, tensorboard_dir, args, last_activation,
                metrics, loss, X_train, Y_train, X_val, Y_val, weights_path=None):
    print("Creating model")
    sys.stdout.flush()
    reset_tensorflow()
    model = get_model(args.model_name,
                      dropout=args.dropout,
                      last_activation=last_activation,
                      activation=args.activation,
                      channels=3 if args.use_depth else None)
    optimizer = args.optimizer
    if optimizer == 'sgd':
        optimizer = SGD(momentum=args.momentum, decay=args.weight_decay)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
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
    parser.add_argument('--weights-path', default=None, type=str)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=0.0001, type=float)
    parser.add_argument('--activation', default='relu')
    parser = add_datagen_args(parser)
    return parser


def pipeline(args):
    ############################# INITIALIZATION #############################
    batch_size = args.batch_size
    shuffle = args.shuffle
    random_seed = args.random_seed

    experiment_name = args.name
    data_path = args.data_path
    log_path = args.log_path

    tensorboard_dir = os.path.join(log_path, 'tensorboard', experiment_name)
    submission_dir = os.path.join(log_path, 'submission')
    submission_path = os.path.join(submission_dir, experiment_name + '_%s.csv')
    model_dir = os.path.join(log_path, './checkpoints')
    model_path = os.path.join(model_dir, experiment_name + '.h5')

    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    train_img_path = os.path.join(train_path, 'images')
    train_mask_path = os.path.join(train_path, 'masks')
    test_img_path = os.path.join(test_path, 'images')

    if not os.path.isdir(submission_dir):
        os.makedirs(submission_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    ############################# GET IMAGE PREPROCESSING INFO #############################
    model_class = get_model_class(args.model_name)
    image_process_func = model_class.get_image_preprocessor()
    image_size = model_class.get_image_size()
    n_channels = model_class.get_number_of_channels()
    if args.use_depth:
        n_channels = 3


    ############################# CREATE TRAIING SET #############################
    train_ids = os.listdir(train_img_path)
    print("Preparing training set")
    sys.stdout.flush()
    images, masks = get_dataset(train_ids, train_img_path, train_mask_path,
                                image_size=image_size,
                                is_test=False,
                                preprocess_func=image_process_func,
                                single_channel=n_channels == 1,
                                use_depth=args.use_depth)
    print("Preparing test set")
    sys.stdout.flush()
    test_ids = os.listdir(test_img_path)
    X_test = get_dataset(test_ids, test_img_path, None,
                         image_size=image_size,
                         is_test=True,
                         preprocess_func=image_process_func,
                         single_channel=n_channels == 1,
                         use_depth=args.use_depth)
    print("Train shape:", images.shape)
    print("Test shape:", X_test.shape)
    sys.stdout.flush()

    ############################# TRAINING #############################
    gc.collect()
    # Prepairing data
    X_train, X_val, Y_train, Y_val = train_test_split(images, masks,
                                                      test_size=args.validation_split,
                                                      shuffle=shuffle,
                                                      random_state=args.random_seed)
    args.validation_split = 0
    train_gen = create_datagen(X_train, Y_train, args,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               random_seed=random_seed)
    val_gen = create_datagen(X_val, Y_val, args,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               random_seed=random_seed)
    train_steps = int(np.ceil(len(X_train) / batch_size))
    val_steps = int(np.ceil(len(X_val) / batch_size))
    gc.collect()

    ############################# Creating model #############################
    print("Stage 1: binary crossentropy loss")
    training_stage(train_gen, val_gen, train_steps, val_steps, model_path,
                   tensorboard_dir, args, "sigmoid", [my_iou_metric], "binary_crossentropy",
                   X_train, Y_train, X_val, Y_val, weights_path=args.weights_path)
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
    print("Making submission")
    make_submission(test_ids, preds_test, submission_path % '_thr', mode="threshold", threshold=threshold_best)
    make_submission(test_ids, preds_test, submission_path % '_round', mode="threshold", threshold=0.0)


if __name__ == "__main__":
    pipeline(create_parser().parse_args())
