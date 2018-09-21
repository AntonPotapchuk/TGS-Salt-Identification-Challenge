from keras import Model
from keras.layers import UpSampling2D, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.applications.inception_resnet_v2 import preprocess_input
from models.inception_resnet_2 import InceptionResNetV2
from models.common import *
from models.model_base import ModelBase


class UnetInceptionResnet2(ModelBase):
    def __init__(self, dropout=0.0, last_activation='sigmoid', activation='relu'):
        super(UnetInceptionResnet2, self).__init__(dropout, last_activation)

    @staticmethod
    def get_image_size():
        return 224

    def _create_model(self, input_shape, dropout=0, last_activation='sigmoid'):
        base_model = InceptionResNetV2(include_top=False, input_shape=input_shape)

        conv1 = base_model.get_layer('activation_3').output
        conv2 = base_model.get_layer('activation_5').output
        conv3 = base_model.get_layer('block35_10_ac').output
        conv4 = base_model.get_layer('block17_20_ac').output
        conv5 = base_model.get_layer('conv_7b_ac').output
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
        conv10 = SpatialDropout2D(0.4)(conv10)
        x = Conv2D(1, (1, 1), activation=last_activation, name="prediction")(conv10)
        model = Model(base_model.input, x)
        return model

    @staticmethod
    def get_image_preprocessor():
        return preprocess_input
