import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def reset_tensorflow():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


def get_model(name, dropout=0.0, last_activation='sigmoid'):
    return get_model_class(name)(dropout, last_activation)


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


reset_tensorflow()
