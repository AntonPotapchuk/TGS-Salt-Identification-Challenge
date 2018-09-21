from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout
from models.model_base import ModelBase
from common.constants import ORIGINAL_IMAGE_SIZE


class UnetBaseline(ModelBase):
    def __init__(self, dropout=0, last_activation='sigmoid', activation='relu'):
        super(UnetBaseline, self).__init__(dropout, last_activation)

    @staticmethod
    def get_image_size():
        return 128

    @staticmethod
    def __conv_block(inp, filters, dropout=0):
        c = Conv2D(filters, (3, 3), activation='relu', padding='same')(inp)
        c = BatchNormalization()(c)
        if dropout > 0:
            c = Dropout(dropout)(c)
        c = Conv2D(filters, (3, 3), activation='relu', padding='same')(c)
        c = BatchNormalization()(c)
        return c

    def _create_model(self, input_shape, dropout=0, last_activation='sigmoid'):
        # Build U-Net model
        inputs = Input(input_shape)
        # Not dropout for the first layer
        c1 = self.__conv_block(inputs, 8, 0)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = self.__conv_block(p1, 16, dropout)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = self.__conv_block(p2, 32, dropout)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = self.__conv_block(p3, 64, dropout)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        # No dropout for the shallow place
        c5 = self.__conv_block(p4, 128, 0)

        u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = self.__conv_block(u6, 64, dropout)

        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = self.__conv_block(u7, 32, dropout)

        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = self.__conv_block(u8, 16, dropout)

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = self.__conv_block(u9, 8, 0)

        outputs = Conv2D(1, (1, 1), activation=last_activation)(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    @staticmethod
    def get_image_preprocessor():
        return None
