import downhill
import numpy as np
import theano
import theano.tensor as TT


def build_rosen(algo):
    x = theano.shared(-3 + np.zeros((2, ), 'f'), name='x')
    return downhill.build(
        algo,
        loss=(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum(),
        params=[x],
        inputs=[],
        updates=(),
        monitors=[('x', x[:-1].sum()), ('y', x[1:].sum())])


def build_factor(algo):
    x = TT.matrix('x')
    u = theano.shared(0.001 + np.zeros((100, 10), 'f'), name='u')
    v = theano.shared(-0.001 + np.zeros((10, 100), 'f'), name='v')
    return downhill.build(
        algo,
        loss=TT.mean(x - TT.dot(u, v)),
        params=[u, v],
        inputs=[x],
        monitors=[
            ('u<1', (u < 1).mean()),
            ('u<-1', (u < -1).mean()),
            ('v<1', (u < 1).mean()),
            ('v<-1', (u < -1).mean()),
        ])
