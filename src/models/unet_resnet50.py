# https://github.com/killthekitten/kaggle-carvana-2017/blob/master/models.py
from keras.engine.training import Model
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import SpatialDropout2D
from keras.layers.merge import concatenate
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import VGG16
from models.common import *
from models.model_base import ModelBase
from models.resnet50_fixed import ResNet50


class UnetResnet50(ModelBase):
    def __init__(self, dropout=0.0, last_activation='sigmoid', activation='relu'):
        super(UnetResnet50, self).__init__(dropout, last_activation)

    @staticmethod
    def get_image_size():
        return 224

    def _create_model(self, input_shape, dropout=0, last_activation='sigmoid'):
        resnet_base = ResNet50(input_shape=input_shape, include_top=False)

        for l in resnet_base.layers:
            l.trainable = True
        conv1 = resnet_base.get_layer("activation_1").output
        conv2 = resnet_base.get_layer("activation_10").output
        conv3 = resnet_base.get_layer("activation_22").output
        conv4 = resnet_base.get_layer("activation_40").output
        conv5 = resnet_base.get_layer("activation_49").output

        up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
        conv6 = conv_block_simple(up6, 256, "conv6_1")
        conv6 = conv_block_simple(conv6, 256, "conv6_2")

        up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
        conv7 = conv_block_simple(up7, 192, "conv7_1")
        conv7 = conv_block_simple(conv7, 192, "conv7_2")

        up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
        conv8 = conv_block_simple(up8, 128, "conv8_1")
        conv8 = conv_block_simple(conv8, 128, "conv8_2")

        up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
        conv9 = conv_block_simple(up9, 64, "conv9_1")
        conv9 = conv_block_simple(conv9, 64, "conv9_2")

        vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
        for l in vgg.layers:
            l.trainable = False
        vgg_first_conv = vgg.get_layer("block1_conv2").output
        up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
        conv10 = conv_block_simple(up10, 32, "conv10_1")
        conv10 = conv_block_simple(conv10, 32, "conv10_2")
        conv10 = SpatialDropout2D(0.2)(conv10)
        x = Conv2D(1, (1, 1), activation=last_activation, name="prediction")(conv10)
        model = Model(resnet_base.input, x)
        return model

    @staticmethod
    def get_image_preprocessor():
        return preprocess_input
