import gc
import os
import sys
import numpy as np

from argparse import ArgumentParser
from keras.models import load_model
from data.loading import get_dataset
from data.datagen import create_datagen
from data.submission import make_submission
from models import get_model
from data.validation import single_model_train_test_split

# TODO: hardcoded!!
ORIGINAL_SIZE = 101


def create_parser():
    from data.datagen import add_argparser_arguments as add_datagen_args
    parser = ArgumentParser(description="Single model pipeline")
    parser.add_argument('--name', help='Experiment name (experiment)', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-name', default='baseline')
    parser.add_argument('--loss', default='binary_crossentropy')
    parser.add_argument('--log-path', default='./')
    parser.add_argument('--random-seed', default=123, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--shuffle', default=True, action='store_true')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--test-size', default=0.3, type=float)
    parser.add_argument('--dropout', help='Fraction of the input units to drop', default=0, type=float)
    parser.add_argument('--early-stopping', help='early-stopping-patience', default=9, type=int)
    parser.add_argument('--reduce-lr-patience', default=3, type=int)
    parser.add_argument('--reduce-lr-alpha', default=0.2, type=float)
    parser = add_datagen_args(parser)
    return parser


# def predict_result(model, x_test, verbose=0): # predict both orginal and reflect x
#     preds_test = np.squeeze(model.predict(x_test, verbose=verbose))
#     preds_test += np.array([ np.fliplr(a) for a in np.squeeze(model.predict(np.array([np.fliplr(x) for x in x_test]), verbose=verbose))])
#     return preds_test / 2.0


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
    submission_path = os.path.join(submission_dir, experiment_name + '.csv')
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

    ############################# Creating model #############################
    sys.stdout.flush()
    print("Creating model...")
    model = get_model(args.model_name, dropout=args.dropout)
    model.compile(optimizer='adam', loss=args.loss, metrics=['mean_prec_iou'])
    image_size = model.get_image_size()
    image_process_func = model.get_image_preprocessor()

    ############################# First data preprocessing #############################
    train_ids, dev_ids = single_model_train_test_split(train_img_path, train_mask_path, args.test_size)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    X_train, Y_train = get_dataset(train_ids, train_img_path, train_mask_path, image_size=image_size, is_test=False,
                                   preprocess_func=image_process_func)
    print('Getting and resizing devel images and masks ... ')
    X_dev, Y_dev = get_dataset(dev_ids, train_img_path, train_mask_path, image_size=image_size, is_test=False,
                               preprocess_func=image_process_func)
    print('Done!')
    print("Train shape:", X_train.shape)
    print("Dev shape:", X_dev.shape)

    if args.validation_split == 0:
        train_gen = create_datagen(X_train, Y_train, args,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   random_seed=random_seed)
    else:
        train_gen, val_gen = create_datagen(X_train, Y_train, args,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    random_seed=random_seed)

    # fits the model on batches with real-time data augmentation:

    ############################# First training #############################
    print("Fitting model without holdoout")
    train_steps_per_epoch = int(len(X_train) * (1 - args.validation_split) / batch_size)
    val_steps_per_epoch = int(len(X_train) * args.validation_split / batch_size)
    train_params = {'train_gen': train_gen,
                    'val_gen': val_gen,
                    'epochs': args.epochs,
                    'shuffle': args.shuffle,
                    'steps_per_epoch': train_steps_per_epoch,
                    'validation_steps': val_steps_per_epoch,
                    'lr_patience': args.reduce_lr_patience,
                    'lr_alpha': args.reduce_lr_alpha,
                    'early_stopping': args.early_stopping,
                    'model_path': model_path,
                    'tensorboard_dir': tensorboard_dir}
    model.fit_generator(**train_params)
    print("Evaluating")
    res = model.evaluate(X_train, Y_train, verbose=0)
    print("Train loss:", res[0])
    print("Train mean IoU:", res[1])
    res = model.evaluate(X_dev, Y_dev, verbose=0)
    print("Test loss:", res[0])
    print("Test mean IoU:", res[1])

    ############################# Second data preprocessing #############################
    X_full = np.concatenate([X_train, X_dev])
    Y_full = np.concatenate([Y_train, Y_dev])
    del X_train, Y_train, X_dev, Y_dev
    gc.collect()

    print("Creating test set")
    test_ids = os.listdir(test_img_path)
    X_test = get_dataset(test_ids, test_img_path, None, image_size=image_size, is_test=True,
                         preprocess_func=image_process_func)

    train_gen, val_gen = create_datagen(X_full, Y_full, args,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        random_seed=random_seed)
    ############################# Second training #############################
    print("Loading best model")
    model.load_weights(model)
    # fits the model on batches with real-time data augmentation:
    print("Fitting model on full dataset")
    train_steps_per_epoch = int(len(X_full) * (1 - args.validation_split) / batch_size)
    val_steps_per_epoch = int(len(X_full) * args.validation_split / batch_size)
    train_params['train_gen'] = train_gen
    train_params['val_gen'] = val_gen
    train_params['steps_per_epoch'] = train_steps_per_epoch
    train_params['validation_steps'] = val_steps_per_epoch
    model.fit_generator(**train_params)
    ############################# Prediction #############################
    # Predict on train, val and test
    print("Loading best model")
    model.load_weights(model_path)
    print("Making predictions")
    preds_test = model.predict(X_test, verbose=1)
    print("Making submission")
    make_submission(test_ids, preds_test, submission_path, original_size=ORIGINAL_SIZE)


if __name__ == "__main__":
    pipeline(create_parser().parse_args())