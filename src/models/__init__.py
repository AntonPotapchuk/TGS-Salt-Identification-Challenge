import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


def get_model(name, dropout=0.0):
    if name == 'baseline':
        from models.unet_baseline_model import UnetBaseline
        return UnetBaseline(dropout)
    if name == 'unet_resnet50':
        from models.unet_resnet50 import UnetResnet50
        return UnetResnet50(dropout)
    if name == 'unet_inception_resnet_v2':
        from models.unet_inception_resnet_v2 import UnetInceptionResnet2
        return UnetInceptionResnet2(dropout)
    raise Exception("Unknown model")
