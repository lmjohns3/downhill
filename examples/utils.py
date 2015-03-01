import climate
import numpy as np

logging = climate.get_logger(__name__)

try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.critical('please install matplotlib to run the examples!')
    raise

try:
    import skdata.mnist
    #import skdata.cifar10
except ImportError:
    logging.critical('please install skdata to run the examples!')
    raise


def load_mnist(labels=False):
    '''Load the MNIST digits dataset.'''
    mnist = skdata.mnist.dataset.MNIST()
    mnist.meta  # trigger download if needed.
    def arr(n, dtype):
        arr = mnist.arrays[n]
        return arr.reshape((len(arr), -1)).astype(dtype)
    train_images = arr('train_images', np.float32) / 128 - 1
    train_labels = arr('train_labels', np.uint8)
    test_images = arr('test_images', np.float32) / 128 - 1
    test_labels = arr('test_labels', np.uint8)
    if labels:
        return ((train_images[:50000], train_labels[:50000, 0]),
                (train_images[50000:], train_labels[50000:, 0]),
                (test_images, test_labels[:, 0]))
    return train_images[:50000], train_images[50000:], test_images


def load_cifar(labels=False):
    cifar = skdata.cifar10.dataset.CIFAR10()
    cifar.meta  # trigger download if needed.
    pixels = cifar._pixels.astype(np.float32).reshape((len(cifar._pixels), -1)) / 128 - 1
    if labels:
        labels = cifar._labels.astype(np.uint8)
        return ((pixels[:40000], labels[:40000, 0]),
                (pixels[40000:50000], labels[40000:50000, 0]),
                (pixels[50000:], labels[50000:, 0]))
    return pixels[:40000], pixels[40000:50000], pixels[50000:]


def plot_images(imgs, loc=111, title=None, channels=1):
    '''Plot an array of images.

    We assume that we are given a matrix of data whose shape is (n*n, s*s*c) --
    that is, there are n^2 images along the first axis of the array, and each
    image is c squares measuring s pixels on a side. Each row of the input will
    be plotted as a sub-region within a single image array containing an n x n
    grid of images.
    '''
    n = int(np.sqrt(len(imgs)))
    assert n * n == len(imgs), 'images array must contain a square number of rows!'
    s = int(np.sqrt(len(imgs[0]) / channels))
    assert s * s == len(imgs[0]) / channels, 'images must be square!'

    img = np.zeros((s * n, s * n, channels), dtype=imgs[0].dtype)
    for i, pix in enumerate(imgs):
        r, c = divmod(i, n)
        img[r * s:(r+1) * s, c * s:(c+1) * s] = pix.reshape((s, s, channels))

    img -= img.min()
    img /= img.max()

    ax = plt.gcf().add_subplot(loc)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.imshow(img.squeeze(), cmap=plt.cm.gray)
    if title:
        ax.set_title(title)

