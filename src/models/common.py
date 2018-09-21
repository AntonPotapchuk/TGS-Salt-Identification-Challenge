from keras.layers import Conv2D, BatchNormalization, Activation


def conv_block_simple(prev_layer, filters, prefix, strides=(1, 1), activation='relu'):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal",
                  strides=strides, name=prefix + "_conv")(prev_layer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv


def conv_block_simple_no_bn(prev_layer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal",
                  strides=strides, name=prefix + "_conv")(prev_layer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv
