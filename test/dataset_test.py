import downhill
import numpy as np
import theano
import theano.tensor as TT


def assert_size(ds, i, expected):
    s = ds._slices[i][0][0]
    assert s.stop - s.start == expected


class TestDataset:
    def test_rng(self):
        ds = downhill.Dataset([np.random.randn(40, 2)], rng=4)
        assert ds.rng.randint(10) == 7
        ds = downhill.Dataset([np.random.randn(40, 2)], rng=np.random.RandomState(4))
        assert ds.rng.randint(10) == 7

    def test_name(self):
        ds = downhill.Dataset([np.random.randn(40, 2)], name='foo')
        assert ds.name == 'foo'
        ds = downhill.Dataset([np.random.randn(40, 2)])
        assert ds.name.startswith('dataset')
        assert ds.name[7:].isdigit()

    def test_batch_size(self):
        ds = downhill.Dataset([np.random.randn(40, 2)], batch_size=10, rng=4)
        assert len(ds._slices) == 4
        assert_size(ds, 0, 10)
        assert_size(ds, 1, 10)
        assert_size(ds, 2, 10)
        assert_size(ds, 3, 10)
        ds = downhill.Dataset([np.random.randn(40, 2)], batch_size=11, rng=4)
        assert len(ds._slices) == 4
        assert_size(ds, 0, 11)
        assert_size(ds, 1, 11)
        assert_size(ds, 2, 7)
        assert_size(ds, 3, 11)

    def test_iteration_size(self):
        def batches_unchanged(previous):
            return all(a == b for a, b in zip(ds._slices, previous))

        ds = downhill.Dataset([np.random.randn(40, 2)],
                              batch_size=5, iteration_size=3)

        previous = list(ds._slices)
        c = sum(1 for _ in ds)
        assert c == 3, 'got {}'.format(c)
        assert ds._index == 3, 'got {}'.format(ds._index)
        assert batches_unchanged(previous)

        previous = list(ds._slices)
        c = sum(1 for _ in ds)
        assert c == 3
        assert ds._index == 6, 'got {}'.format(ds._index)
        assert batches_unchanged(previous)

        previous = list(ds._slices)
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

    def test_shared(self):
        x = theano.shared(np.random.randn(40, 2))
        ds = downhill.Dataset([x], batch_size=10, rng=4)
        assert len(ds._slices) == 4
        assert_size(ds, 0, 10)
        assert_size(ds, 1, 10)
        assert_size(ds, 2, 10)
        assert_size(ds, 3, 10)
        f = list(ds)[0][0]
        assert isinstance(f, TT.TensorVariable), type(f)

    def test_pandas(self):
        import pandas as pd
        x = pd.DataFrame(np.random.randn(40, 2))
        ds = downhill.Dataset([x], batch_size=10, rng=4)
        assert len(ds._slices) == 4
        assert_size(ds, 0, 10)
        assert_size(ds, 1, 10)
        assert_size(ds, 2, 10)
        assert_size(ds, 3, 10)
        f = list(ds)[0][0]
        assert isinstance(f, pd.DataFrame), type(f)

    def test_bad_input_type(self):
        try:
            downhill.Dataset([[1]])
            assert False
        except ValueError:
            pass
