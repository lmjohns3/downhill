import climate
import numpy as np
import theano
import theano.tensor as TT
import downhill

import utils

climate.enable_default_logging()

(t_images, t_labels), (v_images, v_labels), _ = utils.load_mnist(labels=True)

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

utils.plot_images(v.get_value(), 121)
utils.plot_images(np.dot(u.get_value(), v.get_value()), 122)
utils.plt.show()
