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
        monitors=[('x', x[:-1].sum()), ('y', x[1:].sum())]), [[]]


def build_factor(algo):
    x = TT.matrix('x')
    u = theano.shared(np.arange(1000).reshape((100, 10)).astype('f'), name='u')
    v = theano.shared(0.2 + np.zeros((10, 100), 'f'), name='v')
    return downhill.build(
        algo,
        loss=TT.sum(TT.sqr(x - TT.dot(u, v))),
        params=[u, v],
        inputs=[x],
        monitors=[
            ('u<1', (u < 1).mean()),
            ('u<-1', (u < -1).mean()),
            ('v<1', (v < 1).mean()),
            ('v<-1', (v < -1).mean()),
        ]), [np.dot(np.arange(1000).reshape((100, 10)).astype('f'),
                    0.1 + np.zeros((10, 100), 'f'))[None, :, :]]


def assert_progress(opt, train, valid=None, **kwargs):
    mover = opt.iteropt(train, valid=valid, **kwargs)
    train0, valid0 = next(mover)
    train1, valid1 = next(mover)
    assert train1['loss'] < valid0['loss']   # should have made progress!
    assert valid1['loss'] == valid0['loss']  # no new validation occurred
