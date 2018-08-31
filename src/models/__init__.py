import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


def add_argparser_arguments(parser):
    parser.add_argument('--early-stopping', help='early-stopping-patience', default=9, type=int)
    parser.add_argument('--reduce-lr-patience', default=3, type=int)
    parser.add_argument('--reduce-lr-alpha', default=0.2, type=float)
    return parser


def get_model(name, input_shape, dropout=0):
    if name == 'baseline':
        from models.baseline_model import get_model
        return get_model(input_shape, dropout)
    if name == 'unet_resnet34':
        from models.unet_resnet34 import get_model
        return get_model(input_shape, dropout)
    if name == 'unet_resnet50':
        from models.unet_resnet50 import get_model
        return get_model(input_shape, dropout)
    raise Exception("Unknown model")


def get_loss(name):
    if name == 'binary_crossentropy':
        return name
    raise Exception("Unknown loss: " + name)


def get_callbacks(model_path, args, tensorboard_dir = None):
    earlystopper = EarlyStopping(patience=args.early_stopping, verbose=1)
    checkpointer = ModelCheckpoint(model_path, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(factor=args.reduce_lr_alpha, patience=args.reduce_lr_patience)
    callbacks = [earlystopper, checkpointer, reduce_lr]
    if tensorboard_dir != None:
        callbacks.append(TensorBoard(tensorboard_dir))
    return callbacks

