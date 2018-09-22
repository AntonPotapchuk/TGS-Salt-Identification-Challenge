import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


def reset_tensorflow():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


def get_callbacks(model_path, args, tensorboard_dir=None):
    callbacks = []
    if args.early_stopping is not None and args.early_stopping > 0:
        callbacks.append(EarlyStopping(patience=args.early_stopping, verbose=1, monitor="loss"))
    if tensorboard_dir is not None:
        callbacks.append(TensorBoard(tensorboard_dir, write_graph=False))
    if model_path is not None:
        monitor = "loss" if args.save_not_best_only else "val_loss"
        print("Optimize:", monitor)
        callbacks.append(ModelCheckpoint(model_path, save_best_only=True, monitor=monitor))
    if args.reduce_lr_patience is not None and args.reduce_lr_patience > 0:
        callbacks.append(ReduceLROnPlateau(monitor='loss',
                                           factor=args.reduce_lr_alpha,
                                           patience=args.reduce_lr_patience))
    return callbacks



def get_model(name, dropout=0.0, last_activation='sigmoid', activation='relu'):
    return get_model_class(name)(dropout, last_activation, activation)


def get_model_class(name):
    if name == 'baseline':
        from models.unet_baseline_model import UnetBaseline
        return UnetBaseline
    if name == 'unet_resnet50':
        from models.unet_resnet50 import UnetResnet50
        return UnetResnet50
    if name == 'unet_inception_resnet_v2':
        from models.unet_inception_resnet_v2 import UnetInceptionResnet2
        return UnetInceptionResnet2
    if name == 'unet_resnet34':
        from models.unet_resnet34 import UnetResnet34
        return UnetResnet34
    raise ValueError("Not supported model: %s" % name)


reset_tensorflow()
