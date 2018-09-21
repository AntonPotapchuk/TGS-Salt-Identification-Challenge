from keras.layers import add
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.models import Model


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=1, downsample=None, activation='elu'):
    residual = input_tensor
    conv_name_base = 'res_' + stage + '_' + block + '_branch'
    bn_name_base = 'bn_' + stage + '_' + block + '_branch'

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, name=conv_name_base + '2a', padding='same')(input_tensor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation(activation)(x)

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, name=conv_name_base + '2b', padding='same')(x)
    x = BatchNormalization(name=bn_name_base+'2b')(x)

    if downsample is not None:
        residual = downsample

    x = add([x, residual])
    x = Activation(activation)(x)

    return x


def make_layer(input_tensor, kernel_size, filters, blocks, stage, strides=1, activation='elu'):
    downsample = None
    if strides != 1:
        downsample = Conv2D(filters, kernel_size=(1, 1), strides=strides, name="res_"+stage+"_1_branch_1")(input_tensor)
        downsample = BatchNormalization(name="bn_"+stage+"_1_branch_1")(downsample)

    x = conv_block(input_tensor, kernel_size, filters, stage, '1', strides, downsample, activation=activation)
    for i in range(1, blocks):
        x = conv_block(x, kernel_size, filters, stage, str(i + 1), activation=activation)
    return x


def ResNet34(include_top=True, input_shape=None,
             pooling=None,
             classes=1000,
             activation='elu'):

    img_input = Input(shape=input_shape)

    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation(activation)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = make_layer(x, (3, 3), 64, 3, 'a', activation=activation)
    x = make_layer(x, (3, 3), 128, 4, 'b', strides=2, activation=activation)
    x = make_layer(x, (3, 3), 256, 6, 'c', strides=2, activation=activation)
    x = make_layer(x, (3, 3), 512, 3, 'd', strides=2, activation=activation)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Create model.
    model = Model(img_input, x, name='resnet50')
    return model
