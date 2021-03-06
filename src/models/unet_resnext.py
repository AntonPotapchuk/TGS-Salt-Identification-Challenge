from keras import Model
from keras.layers import concatenate, UpSampling2D, Conv2D

from models.common import conv_block_simple
from models.model_base import ModelBase
from models.se_resnext import SEResNextImageNet


class UnetResnext(ModelBase):
    def __init__(self, dropout=0.0, last_activation='sigmoid', activation='relu', channels=None):
        super(UnetResnext, self).__init__(dropout, last_activation, channels)

    @staticmethod
    def get_image_size():
        return 224

    def _create_model(self, input_shape, dropout=0, last_activation='sigmoid'):
        base_model = SEResNextImageNet(include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = True

        conv1 = base_model.get_layer('leaky_re_lu_1').output    # 112x112x64
        conv2 = base_model.get_layer('leaky_re_lu_11').output    # 56x56x64
        conv3 = base_model.get_layer('leaky_re_lu_23').output   # 28x28x128
        conv4 = base_model.get_layer('leaky_re_lu_41').output   # 14x14x256
        conv5 = base_model.get_layer('leaky_re_lu_49').output      # 7x7x512
        up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
        conv6 = conv_block_simple(up6, 256, "conv6_1")
        conv6 = conv_block_simple(conv6, 256, "conv6_2")

        up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
        conv7 = conv_block_simple(up7, 256, "conv7_1")
        conv7 = conv_block_simple(conv7, 256, "conv7_2")

        up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
        conv8 = conv_block_simple(up8, 128, "conv8_1")
        conv8 = conv_block_simple(conv8, 128, "conv8_2")

        up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
        conv9 = conv_block_simple(up9, 64, "conv9_1")
        conv9 = conv_block_simple(conv9, 64, "conv9_2")

        up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=-1)
        conv10 = conv_block_simple(up10, 48, "conv10_1")
        conv10 = conv_block_simple(conv10, 32, "conv10_2")
        # conv10 = SpatialDropout2D(0.4)(conv10)
        x = Conv2D(1, (1, 1), activation=last_activation, name="prediction")(conv10)
        model = Model(base_model.input, x)
        return model

    @staticmethod
    def get_image_preprocessor():
        return lambda img: img / 255.

    @staticmethod
    def get_number_of_channels():
        return 1
