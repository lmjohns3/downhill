import climate
import numpy as np
import theano
import theano.tensor as TT
import theanopt

import utils

climate.enable_default_logging()

train, valid, _ = utils.load_mnist()

N = 11
K = 9
B = 784

x = TT.matrix('x')

u = theano.shared(np.random.randn(N * N, K * K).astype('f'), name='u')
v = theano.shared(np.random.randn(K * K, B).astype('f'), name='v')

err = TT.sqr(x - TT.dot(u, v))

theanopt.minimize(
    loss=err.mean() + 0.1 * abs(u).mean() + 0.01 * (v * v).mean(),
    params=[u, v],
    inputs=[x],
    train=train,
    valid=valid,
    batch_size=N * N,
    monitors=[
        ('u<0.1', 100 * (abs(u) < 0.1).mean()),
        ('v<0.1', 100 * (abs(v) < 0.1).mean()),
    ],
)

utils.plot_images(v.get_value(), 121)
utils.plot_images(np.dot(u.get_value(), v.get_value()), 122)
utils.plt.show()
