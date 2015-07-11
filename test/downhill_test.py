import downhill
import numpy as np
import theano


class TestMinimize:
    def test_minimize(self):
        x = theano.shared(-3 + np.zeros((2, ), 'f'), name='x')
        data = downhill.Dataset(np.zeros((1, 1)), batch_size=1)
        data._batches = [[]]
        downhill.minimize(
            (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum(),
            data,
            algo='nag',
            learning_rate=0.001,
            momentum=0.9,
            patience=1,
            min_improvement=0.1,
            max_gradient_norm=1,
        )
        assert np.allclose(x.get_value(), [1, 1]), x.get_value()
