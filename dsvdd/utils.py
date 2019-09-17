from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np


__all__ = ['plot_most_normal_and_abnormal_images', 'task']


@contextmanager
def task(_=''):
    yield


def flatten_image_list(images, show_shape) -> np.ndarray:
    """

    :param images:
    :param tuple show_shape:
    :return:
    """
    N = np.prod(show_shape)

    if isinstance(images, list):
        images = np.array(images)

    for i in range(len(images.shape)):  # find axis.
        if N == np.prod(images.shape[:i]):
            img_shape = images.shape[i:]
            new_shape = (N,) + img_shape
            return np.reshape(images, new_shape)

    else:
        raise ValueError('Cannot distinguish images. imgs shape: %s, show_shape: %s' % (images.shape, show_shape))


def get_shape(image):
    shape_ = image.shape[-3:]
    if len(shape_) <= 1:
        raise ValueError('Unexpected shape: {}'.format(shape_))

    elif len(shape_) == 2:
        H, W = shape_
        return H, W, 1

    elif len(shape_) == 3:
        H, W, C = shape_
        if C in [1, 3]:
            return H, W, C
        else:
            raise ValueError('Unexpected shape: {}'.format(shape_))

    else:
        raise ValueError('Unexpected shape: {}'.format(shape_))


def merge_image(images, show_shape, order='row'):
    images = flatten_image_list(images, show_shape)
    H, W, C = get_shape(images)
    I, J = show_shape
    result = np.zeros((I * H, J * W, C), dtype=images.dtype)

    for k, img in enumerate(images):
        if order.lower().startswith('row'):
            i = k // J
            j = k % J
        else:
            i = k % I
            j = k // I

        result[i * H: (i + 1) * H, j * W: (j + 1) * W] = img

    return result


def plot_most_normal_and_abnormal_images(X_test, score):
    fig, axes = plt.subplots(nrows=2)
    fig.set_size_inches((5, 5))
    inds = np.argsort(score)

    image1 = merge_image(X_test[inds[:10]], (2, 5))
    axes[0].imshow(np.squeeze(image1))
    axes[0].set_title('Most normal images')
    axes[0].set_axis_off()

    image2 = merge_image(X_test[inds[-10:]], (2, 5))
    axes[1].imshow(np.squeeze(image2))
    axes[1].set_title('Most abnormal images')
    axes[1].set_axis_off()
