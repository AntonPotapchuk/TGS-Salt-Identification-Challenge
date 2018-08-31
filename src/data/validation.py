import os
import numpy as np

# TODO: Let's remove train_mask_path. Less save, but we know that ids are same
def single_model_train_test_split(train_img_path, train_mask_path, test_size=0.3):
    # TODO: separate function?
    images_ids = os.listdir(train_img_path)
    masks_ids = os.listdir(train_mask_path)
    # sort to avoid any OS specific dependency
    images_ids = np.sort(images_ids)
    masks_ids = np.sort(masks_ids)
    # make sure masks are same as images
    assert (images_ids == masks_ids).all()

    # train-dev split (my vocabulary: train, validation, development, test)
    # using Keras API later to create validation set on the train data!
    # development set would be an equivalent for the test, but for developement!
    train_size = int((1. - test_size) * len(images_ids))
    np.random.seed(73)
    picked = np.random.choice(np.arange(len(images_ids)), train_size, replace=False)
    mask = np.zeros(len(images_ids))
    mask[picked] = 1

    # make sure the masking was ok
    assert len(mask) == len(images_ids)
    assert np.sum(mask) == len(picked)

    mask = np.array(mask, dtype=np.bool)
    train_ids = np.array(images_ids)[mask]
    dev_ids = np.array(masks_ids)[~mask]

    # make sure everything went right
    assert train_size == len(train_ids)
    assert len(dev_ids) == (len(images_ids) - train_size)

    return train_ids, dev_ids