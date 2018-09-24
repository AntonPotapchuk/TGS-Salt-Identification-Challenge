from keras import Input, Model
from keras.layers import Activation, BatchNormalization, Conv2D, add, MaxPooling2D, GlobalAveragePooling2D, Dense, \
    GlobalMaxPooling2D
from models.se import scse_block


def SEResNet34(input_shape=None,
               depth=[3, 4, 6, 3],
               filters=[64, 128, 256, 512],
               include_top=True,
               pooling=None,
               classes=1000,
               activation='relu'):
    """ Instantiate the Squeeze and Excite ResNet architecture. Note that ,
            when using TensorFlow for best performance you should set
            `image_data_format="channels_last"` in your Keras config
            at ~/.keras/keras.json.
            The model are compatible with both
            TensorFlow and Theano. The dimension ordering
            convention used by the model is the one
            specified in your Keras config file.
            # Arguments
                initial_conv_filters: number of features for the initial convolution
                depth: number or layers in the each block, defined as a list.
                    ResNet-50  = [3, 4, 6, 3]
                    ResNet-101 = [3, 6, 23, 3]
                    ResNet-152 = [3, 8, 36, 3]
                filter: number of filters per block, defined as a list.
                    filters = [64, 128, 256, 512
                include_top: whether to include the fully-connected
                    layer at the top of the network.
                weights: `None` (random initialization) or `imagenet` (trained
                    on ImageNet)
                input_shape: optional shape tuple, only to be specified
                    if `include_top` is False (otherwise the input shape
                    has to be `(224, 224, 3)` (with `tf` dim ordering)
                    or `(3, 224, 224)` (with `th` dim ordering).
                    It should have exactly 3 inputs channels,
                    and width and height should be no smaller than 8.
                    E.g. `(200, 200, 3)` would be one valid value.
                pooling: Optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                        the 4D tensor output of the
                        last convolutional layer.
                    - `avg` means that global average pooling
                        will be applied to the output of the
                        last convolutional layer, and thus
                        the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                        be applied.
                classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
            # Returns
                A Keras model instance.
            """
    assert len(depth) == len(filters), "The length of filter increment list must match the length " \
                                       "of the depth list."
    img_input = Input(shape=input_shape)
    x = _create_se_resnet(classes, img_input, include_top, filters, depth, pooling, activation)
    # Create model.
    model = Model(img_input, x, name='resnext')
    return model


def _resnet_block(input, filters, strides=(1, 1), activation='relu'):
    ''' Adds a pre-activation resnet block without bottleneck layers
    Args:
        input: input tensor
        filters: number of output filters
        strides: strides of the convolution layer
    Returns: a keras tensor
    '''
    init = input

    x = BatchNormalization()(input)
    x = Activation(activation)(x)

    if strides != (1, 1) or init.shape[-1] != filters:
        init = Conv2D(filters, (1, 1), padding='same', strides=strides)(x)

    x = Conv2D(filters, (3, 3), padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(filters, (3, 3), padding='same')(x)

    m = add([x, init])
    return m


def _create_se_resnet(classes, img_input, include_top, filters,
                      depth, pooling, activation):
    '''Creates a SE ResNet model with specified parameters
    Args:
        include_top: Flag to include the last dense layer
        filters: number of filters per block, defined as a list.
            filters = [64, 128, 256, 512
        depth: number or layers in the each block, defined as a list.
            ResNet-50  = [3, 4, 6, 3]
            ResNet-101 = [3, 6, 23, 3]
            ResNet-152 = [3, 8, 36, 3]
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    '''
    N = list(depth)

    # block 1 (initial conv block)
    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2))(img_input)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # block 2 (projection block)
    for i in range(N[0]):
        x = _resnet_block(x, filters[0], activation=activation)
        # squeeze and excite block
    x = scse_block(x, activation=activation)

    # block 3 - N
    for k in range(1, len(N)):
        x = _resnet_block(x, filters[k], strides=(2, 2), activation=activation)
        for i in range(N[k] - 1):
            x = _resnet_block(x, filters[k], activation=activation)
            # squeeze and excite block
        x = scse_block(x, activation=activation)

    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, activation='softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    return x
