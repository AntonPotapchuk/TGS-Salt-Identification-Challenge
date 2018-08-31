from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout

def __conv_block(inp, filters, dropout=0):
    c = Conv2D(filters, (3, 3), activation='relu', padding='same')(inp)
    c = BatchNormalization()(c)
    if dropout > 0:
        c = Dropout(dropout)(c)
    c = Conv2D(filters, (3, 3), activation='relu', padding='same')(c)
    c = BatchNormalization()(c)
    return c


def get_model(input_shape, dropout=0):
    # Build U-Net model
    inputs = Input(input_shape)
    # Not dropout for the first layer
    c1 = __conv_block(inputs, 8, 0)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = __conv_block(p1, 16, dropout)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = __conv_block(p2, 32, dropout)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = __conv_block(p3, 64, dropout)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    # No dropoout for the shallow place
    c5 = __conv_block(p4, 128, 0)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = __conv_block(u6, 64, dropout)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = __conv_block(u7, 32, dropout)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = __conv_block(u8, 16, dropout)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = __conv_block(u9, 8, 0)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model