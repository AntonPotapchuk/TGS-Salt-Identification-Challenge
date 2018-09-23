from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute, Activation, Conv2D, add
from keras import backend as K


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def scse_block(inp, reduction=8, activation='relu'):
    filters = inp._keras_shape[-1]
    se_shape = (1, 1, filters)

    # Returns a new tensor with the same data as the self tensor but of a different size.
    chn_se = GlobalAveragePooling2D()(inp)
    chn_se = Reshape(se_shape)(chn_se)
    chn_se = Dense(filters // reduction, activation=activation)(chn_se)
    chn_se = Dense(filters, activation='sigmoid')(chn_se)
    chn_se = multiply([inp, chn_se])

    spa_se = Conv2D(filters=1, kernel_size=1, strides=1, padding='valid', use_bias=False)(inp)
    spa_se = Activation('sigmoid')(spa_se)
    spa_se = multiply([inp, spa_se])

    return add([chn_se, spa_se])
