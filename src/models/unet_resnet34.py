from keras.layers import Conv2D, UpSampling2D
from keras.layers.merge import concatenate
from keras.models import Model
from models.common import conv_block_simple
from models.model_base import ModelBase
from models.resnet34 import ResNet34


def preprocess_image(img):
    return img / 256.


class UnetResnet34(ModelBase):
    def __init__(self, dropout=0.0, last_activation='sigmoid'):
        super(UnetResnet34, self).__init__(dropout, last_activation)

    @staticmethod
    def get_image_size():
        return 224

    def _create_model(self, input_shape, dropout=0, last_activation='sigmoid'):
        resnet_base = ResNet34(input_shape=input_shape, include_top=False)

        conv1 = resnet_base.get_layer("elu_1").output           # 112x112x64
        conv2 = resnet_base.get_layer("elu_7").output           # 56x56x64
        conv3 = resnet_base.get_layer("elu_15").output          # 28x28x128
        conv4 = resnet_base.get_layer("elu_27").output          # 14x14x256
        conv5 = resnet_base.get_layer("elu_33").output          # 7x7x512

        up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
        conv6 = conv_block_simple(up6, 256, "conv6_1", activation='elu')
        conv6 = conv_block_simple(conv6, 256, "conv6_2", activation='elu')

        up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
        conv7 = conv_block_simple(up7, 192, "conv7_1", activation='elu')
        conv7 = conv_block_simple(conv7, 192, "conv7_2", activation='elu')

        up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
        conv8 = conv_block_simple(up8, 128, "conv8_1", activation='elu')
        conv8 = conv_block_simple(conv8, 128, "conv8_2", activation='elu')

        up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
        conv9 = conv_block_simple(up9, 64, "conv9_1", activation='elu')
        conv9 = conv_block_simple(conv9, 64, "conv9_2", activation='elu')


        up10 = concatenate([UpSampling2D()(conv9), resnet_base.input], axis=-1)
        conv10 = conv_block_simple(up10, 64, "conv10_1", activation='elu')
        conv10 = conv_block_simple(conv10, 64, "conv10_2", activation='elu')
        # conv10 = SpatialDropout2D(0.2)(conv10)
        x = Conv2D(1, (1, 1), activation=last_activation, name="prediction")(conv10)
        model = Model(resnet_base.input, x)
        return model


    @staticmethod
    def get_image_preprocessor():
        return preprocess_image

    def get_number_of_channels(self):
        return 1
