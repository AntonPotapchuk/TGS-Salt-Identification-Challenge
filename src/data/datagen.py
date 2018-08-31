from keras.preprocessing.image import ImageDataGenerator


def add_argparser_arguments(parser):
    parser.add_argument('--featurewise-center', default=False, action='store_true')
    parser.add_argument('--samplewise-center', default=False, action='store_true')
    parser.add_argument('--featurewise-norm', default=False, action='store_true')
    parser.add_argument('--samplewise-norm', default=False, action='store_true')
    parser.add_argument('--rotation', default=0, type=int)
    parser.add_argument('--width-shift', default=0, type=float)
    parser.add_argument('--height-shift', default=0, type=float)
    parser.add_argument('--zoom-range', default=0, type=float)
    parser.add_argument('--horizontal-flip', default=False, action='store_true')
    parser.add_argument('--validation-split', default=0.2, type=float)
    return parser


def create_datagen(X, y, args, batch_size=32, shuffle=False, random_seed=123):
    data_gen_args = dict(featurewise_center=args.featurewise_center,
                                 featurewise_std_normalization=args.featurewise_norm,
                                 samplewise_center=args.samplewise_center,
                                 samplewise_std_normalization=args.samplewise_norm,
                                 rotation_range=args.rotation,
                                 width_shift_range=args.width_shift,
                                 height_shift_range=args.height_shift,
                                 zoom_range=args.zoom_range,
                                 horizontal_flip=args.horizontal_flip,
                                 validation_split=args.validation_split,
                                 fill_mode='constant')
    # https: // github.com / NVIDIA / keras / blob / master / docs / templates / preprocessing / image.md
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    image_datagen.fit(X, augment=True, seed=random_seed)
    mask_datagen.fit(y, augment=True, seed=random_seed)

    if args.validation_split == 0:
        image_generator = image_datagen.flow(X, batch_size=batch_size, shuffle=shuffle, seed=random_seed)
        mask_generator = mask_datagen.flow(y, batch_size=batch_size, shuffle=shuffle, seed=random_seed)
        return zip(image_generator, mask_generator)

    image_generator_train = image_datagen.flow(X, batch_size=batch_size, shuffle=shuffle, seed=random_seed, subset='training')
    image_generator_test = image_datagen.flow(X, batch_size=batch_size, shuffle=shuffle, seed=random_seed, subset='validation')
    mask_generator_train = mask_datagen.flow(y, batch_size=batch_size, shuffle=shuffle, seed=random_seed, subset='training')
    mask_generator_test = mask_datagen.flow(y, batch_size=batch_size, shuffle=shuffle, seed=random_seed, subset='validation')

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator_train, mask_generator_train)
    val_generator = zip(image_generator_test, mask_generator_test)
    return train_generator, val_generator
