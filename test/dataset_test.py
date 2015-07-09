import downhill
import numpy as np


class TestDataset:
    def test_name(self):
        ds = downhill.Dataset([np.random.randn(40, 2)], name='foo')
        assert ds.name == 'foo'
        ds = downhill.Dataset([np.random.randn(40, 2)])
        assert ds.name.startswith('dataset')
        assert ds.name[7:].isdigit()

    def test_batch_size(self):
        ds = downhill.Dataset([np.random.randn(40, 2)], batch_size=10, rng=4)
        assert len(ds._batches) == 4
        assert ds._batches[0][0].shape == (10, 2)
        assert ds._batches[1][0].shape == (10, 2)
        assert ds._batches[2][0].shape == (10, 2)
        assert ds._batches[3][0].shape == (10, 2)
        ds = downhill.Dataset([np.random.randn(40, 2)], batch_size=11, rng=4)
        assert len(ds._batches) == 4
        assert ds._batches[0][0].shape == (11, 2)
        assert ds._batches[1][0].shape == (11, 2)
        assert ds._batches[2][0].shape == (7, 2)
        assert ds._batches[3][0].shape == (11, 2)

    def test_iteration_size(self):
        def batches_unchanged(previous):
            return all(np.allclose(a, b) for a, b in zip(ds._batches, previous))

        ds = downhill.Dataset([np.random.randn(40, 2)],
                              batch_size=5, iteration_size=3)

        previous = list(ds._batches)
        c = sum(1 for _ in ds)
        assert c == 3, 'got {}'.format(c)
        assert ds._index == 3, 'got {}'.format(ds._index)
        assert batches_unchanged(previous)

        previous = list(ds._batches)
        c = sum(1 for _ in ds)
        assert c == 3
        assert ds._index == 6, 'got {}'.format(ds._index)
        assert batches_unchanged(previous)

        previous = list(ds._batches)
        c = sum(1 for _ in ds)
        assert c == 3
        assert ds._index == 1, 'got {}'.format(ds._index)
        assert not batches_unchanged(previous)

    def test_callable(self):
        def batches():
            return 'hello'
        ds = downhill.Dataset(batches, iteration_size=10)
        assert list(ds) == ['hello'] * 10

    def test_callable_length(self):
        class Batches:
            called = 0
            def __call__(self):
                self.called += 1
                return 'hello'
            def __len__(self):
                return 10
        batches = Batches()
        ds = downhill.Dataset(batches, iteration_size=10)
        assert list(ds) == ['hello'] * 10
        assert batches.called == 10

