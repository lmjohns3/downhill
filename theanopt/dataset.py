# -*- coding: utf-8 -*-

r'''This module contains a class for handling batched datasets.

In many optimization tasks, parameters must be updated by optimizing them with
respect to estimates of a loss function. The loss function for many problems is
estimated using a set of data that we have measured.
'''

import climate
import collections
import numpy.random as rng

logging = climate.get_logger(__name__)


class Dataset:
    '''This class handles batching and shuffling a dataset.

    In ``theanopt``, losses are optimized using sets of data collected from the
    problem that generated the loss.

    During optimization, data are grouped into "mini-batches"---that is, chunks
    that are larger than 1 sample and smaller than the entire set of samples;
    typically the size of a mini-batch is between 10 and 100, but the specific
    setting can be varied depending on your model, hardware, dataset, and so
    forth. These mini-batches must be presented to the optimization algorithm in
    pseudo-random order to match the underlying stochasticity assumptions of
    many optimization algorithms. This class handles the process of grouping
    data into mini-batches as well as iterating and shuffling these mini-batches
    dynamically as the dataset is consumed by the optimization algorithm.

    For many tasks, a dataset is obtained as a large block of sample data, which
    in Python is normally assembled as a ``numpy`` ndarray. To use this class on
    such a dataset, just pass in a ``numpy`` array. If multiple inputs are
    required to compute your loss, pass a tuple or list of ``numpy`` arrays; the
    input arrays should have the same size along one axis, so that they all can
    be split into mini-batches.

    There are some cases when a suitable set of training data would be
    prohibitively expensive to assemble in memory as a single ``numpy`` array.
    To handle these cases, this class can also handle a dataset that is provided
    via a Python callable. For more information on using callables to provide
    data to your model, see :ref:`quickstart-using-callables`.

    Parameters
    ----------
    inputs : ndarray, tuple, list, or callable
        One or more sets of data.

        If this parameter is callable, then mini-batches will be obtained by
        calling the callable with no arguments; the callable is expected to
        return a tuple of ndarrays that will be suitable for optimizing a loss.

        If this parameter is an ndarray, it is assumed to contain data for
        computing the loss, with individual data samples arranged along axis N
        (defaults to 0; see ``axis`` parameter below). If this parameter is a
        list or tuple of ndarrays, they are also assumed to contain data for
        computing the loss; the length of this tuple or list should match the
        number of inputs required by the loss computation. If multiple ndarrays
        are provided, their lengths along axis N must match.
    name : str, optional
        A string that is used to describe this dataset. Usually something like
        'test' or 'train'.
    batch_size : int, optional
        The size of the mini-batches to create from the data sequences. Defaults
        to 32.
    iteration_size : int, optional
        The number of batches to yield for each call to iterate(). Defaults to
        the length of the data divided by batch_size. If the dataset is a
        callable, then the number is len(callable). If callable has no length,
        then the number is set to 100.
    axis : int, optional
        The axis along which to split the data arrays, if the first parameter is
        given as one or more ndarrays. If not provided, defaults to 0.
    '''

    count = 0

    def __init__(self, inputs, name=None, batch_size=32, iteration_size=None, axis=0):
        '''Create a minibatch dataset from data arrays or a callable.'''
        self.name = name or 'dataset{}'.format(Dataset.count)
        Dataset.count += 1
        self.batch_size = batch_size
        self.iteration_size = iteration_size

        self.batches = []

        if isinstance(inputs, collections.Callable):
            self._init_callable(inputs)
        else:
            self._init_arrays(inputs, axis)

    def _init_callable(self, inputs):
        self.batches = inputs
        if not self.iteration_size:
            try:
                self.iteration_size = len(inputs)
            except TypeError: # has no len
                self.iteration_size = 100
        logging.info('%s: %d mini-batches from callable',
                     self.name, self.iteration_size)

    def _init_arrays(self, inputs, axis=0):
        self._index = 0  # index for iteration.

        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs, )

        L = inputs[0].shape[axis]
        assert all(L == x.shape[axis] for x in inputs), \
            'shapes do not match along axis {}: {}'.format(
                axis, '; '.join(str(x.shape) for x in inputs))

        for i in range(0, L, self.batch_size):
            batch = []
            for x in inputs:
                slices = [slice(None) for _ in x.shape]
                slices[axis] = slice(i, i + self.batch_size)
                b = x[tuple(slices)]
                if b.shape[axis] != self.batch_size:
                    break
                batch.append(b)
            else:
                self.batches.append(batch)

        self.shuffle()

        if not self.iteration_size:
            self.iteration_size = len(self.batches)

        logging.info('%s: %d of %d mini-batches of %s',
                     self.name,
                     self.iteration_size,
                     len(self.batches),
                     '; '.join(str(x.shape) for x in self.batches[0]))

    def __iter__(self):
        return self.iterate(True)

    def shuffle(self):
        rng.shuffle(self.batches)

    def iterate(self, update=True):
        return self._iter_callable() \
            if callable(self.batches) \
            else self._iter_batches(update)

    def _iter_batches(self, update=True):
        k = len(self.batches)
        for _ in range(self.iteration_size):
            self._index += 1
            yield self.batches[self._index % k]
        if update:
            self.update()

    def _iter_callable(self):
        for _ in range(self.iteration_size):
            yield self.batches()

    def update(self):
        if self._index >= len(self.batches):
            self.shuffle()
            self._index = 0
