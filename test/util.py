import downhill
import numpy as np
import theano
import theano.tensor as TT


def build_rosen(algo):
    x = theano.shared(-3 + np.zeros((2, ), 'f'), name='x')
    return downhill.build(
        algo,
        loss=(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum(),
        monitors=[('x', x[:-1].sum()), ('y', x[1:].sum())]), [[]]


def build_factor(algo):
    a = np.arange(1000).reshape((100, 10)).astype('f')
    b = 0.1 + np.zeros((10, 100), 'f')

    x = TT.matrix('x')
    u = theano.shared(a, name='u')
    v = theano.shared(0.1 + b, name='v')
    return downhill.build(
        algo,
        loss=TT.sum(TT.sqr(x - TT.dot(u, v))),
        monitors=[
            ('u<1', (u < 1).mean()),
            ('u<-1', (u < -1).mean()),
            ('v<1', (v < 1).mean()),
            ('v<-1', (v < -1).mean()),
        ]), [[np.dot(a, b) + np.random.randn(100, 100).astype('f')]
             for _ in range(10)]


def assert_progress(opt, train, valid=None, **kwargs):
    mover = opt.iterate(train, valid=valid, **kwargs)
    train0, valid0 = next(mover)
    train1, valid1 = next(mover)
    assert train1['loss'] < valid0['loss']   # should have made progress!
    assert valid1['loss'] == valid0['loss']  # no new validation occurred
