import climate
import numpy as np
import theano
import theano.tensor as TT
import downhill

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

climate.enable_default_logging()


def load_mnist():
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
    return ((train_images[:50000], train_labels[:50000, 0]),
            (train_images[50000:], train_labels[50000:, 0]))


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


(t_images, t_labels), (v_images, v_labels) = load_mnist()

# construct training/validation sets consisting of the fours.
train = t_images[t_labels == 4]
valid = v_images[v_labels == 4]

N = 20
K = 20
B = 784

x = TT.matrix('x')

u = theano.shared(np.random.randn(N * N, K * K).astype('f'), name='u')
v = theano.shared(np.random.randn(K * K, B).astype('f'), name='v')

err = TT.sqr(x - TT.dot(u, v)).mean()

downhill.minimize(
    loss=err + 100 * (0.01 * abs(u).mean() + (v * v).mean()),
    params=[u, v],
    inputs=[x],
    train=train,
    valid=valid,
    batch_size=N * N,
    monitor_gradients=True,
    monitors=[
        ('err', err),
        ('u<-0.5', (u < -0.5).mean()),
        ('u<-0.1', (u < -0.1).mean()),
        ('u<0.1', (u < 0.1).mean()),
        ('u<0.5', (u < 0.5).mean()),
    ],
    algo='sgd',
    max_gradient_clip=1,
    learning_rate=0.5,
    momentum=0.9,
    patience=3,
    min_improvement=0.1,
)

plot_images(v.get_value(), 121)
plot_images(np.dot(u.get_value(), v.get_value()), 122)
plt.show()
