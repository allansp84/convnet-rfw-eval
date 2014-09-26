# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

import os

import numpy as np
from scipy import misc


class NotAllowedMethod(Exception):
    """
    Helper class to catch calls to methods not allowed in this repository.
    """
    pass


def get_folders_recursively(path, type):
    """
    Helper function to recursively retrieve all folders containing files of
    type <type> in a given path.
    """

    folders = []

    for root, subFolders, files in os.walk(path):
        for file in files:
            if file[-len(type):] == type:
                folders += [os.path.relpath(root, path)]
                break

    return folders


def load_imgs(fnames, out_shape=None, dtype='uint8',
              flatten=True,  minmax_norm=False):

    if minmax_norm:
        assert ('float' in dtype)

    if flatten:
        n_channels = 1
    else:
        n_channels = 3

    if out_shape == None:
        # -- read first image to retrieve output shape
        out_shape = misc.imread(fnames[0], flatten).shape[:2]
        # -- check later if all images in the dataset have the same shape
        img_resize = False
    else:
        assert len(out_shape) == 2
        img_resize = True

    n_imgs = len(fnames)
    img_set = np.empty((n_imgs,) + out_shape + (n_channels,), dtype=dtype)

    for i, fname in enumerate(fnames):

        arr = misc.imread(fname, flatten)

        if img_resize:
            # -- resize image keeping its aspect ratio and best fitting it to
            #    the desired output
            in_shape = arr.shape[:2]
            resize_shape = tuple((np.array(in_shape) /
                max(np.array(in_shape) / np.array(out_shape,
                dtype=np.float32))).round().astype(np.int))

            arr = misc.imresize(arr, resize_shape).astype(dtype)

            # -- pad the channel mean value when necessary
            pad_size = np.array(out_shape) - np.array(arr.shape)
            assert pad_size.min() == 0

            if pad_size.max() > 0:
                pad_size = pad_size / 2.
                pad_size = ((np.ceil(pad_size[0]), np.floor(pad_size[0])),
                            (np.ceil(pad_size[1]), np.floor(pad_size[1])))

                if not flatten:
                    pad_size += ((0,0),)

                img_mean = int(arr.mean().round())
                arr = np.pad(arr, pad_size, 'constant',
                            constant_values=img_mean)

        if flatten:
            arr.shape = arr.shape + (1,)

        assert arr.shape[:2] == out_shape

        if minmax_norm:
            arr -= arr.min()
            arr /= arr.max()

        img_set[i] = arr

    return img_set
