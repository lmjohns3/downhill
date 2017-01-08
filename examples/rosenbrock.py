'''Helper functions for rosenbrock optimization examples.'''

import downhill
import numpy as np
import theano

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

COLORS = ('#d62728 #1f77b4 #2ca02c #9467bd #ff7f0e '
          '#e377c2 #8c564b #bcbd22 #7f7f7f #17becf').split()


def build(algo, init):
    '''Build and return an optimizer for the rosenbrock function.

    In downhill, an optimizer can be constructed using the build() top-level
    function. This function requires several Theano quantities such as the loss
    being optimized and the parameters to update during optimization.
    '''
    x = theano.shared(np.array(init, 'f'), name='x')
    n = 0.1 * RandomStreams().normal((len(init) - 1, ))
    monitors = []
    if len(init) == 2:
        # this gives us access to the x and y locations during optimization.
        monitors.extend([('x', x[:-1].sum()), ('y', x[1:].sum())])
    return downhill.build(
        algo,
        loss=(n + 100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum(),
        params=[x],
        monitors=monitors,
        monitor_gradients=True)


def build_and_trace(algo, init, limit=100, **kwargs):
    '''Run an optimizer on the rosenbrock function. Return xs, ys, and losses.

    In downhill, optimization algorithms can be iterated over to progressively
    minimize the loss. At each iteration, the optimizer yields a dictionary of
    monitor values that were computed during that iteration. Here we build an
    optimizer and then run it for a fixed number of iterations.
    '''
    kw = dict(min_improvement=0, patience=0, max_gradient_norm=100)
    kw.update(kwargs)
    xs, ys, loss = [], [], []
    for tm, _ in build(algo, init).iterate([[]], **kw):
        if len(init) == 2:
            xs.append(tm['x'])
            ys.append(tm['y'])
        loss.append(tm['loss'])
        if len(loss) == limit:
            break
    # Return the optimization up to any failure of patience.
    return xs[:-9], ys[:-9], loss[-9]


def test(algos, n=10, init=[-1.1, 0], limit=100):
    '''Run several optimizers for comparison.

    Each optimizer is run a fixed number of times with random hyperparameter
    values, and the results are yielded back to the caller (often stored in a
    dictionary).

    Returns
    -------
    results : sequence of (key, value) pairs
        A sequence of results from running tests. Each result contains a "key"
        that describes the test run and a "value" that contains the results from
        the run. The key is a tuple containing (a) the algorithm, (b) the
        learning rate, (c) the momentum, (d) the RMS halflife, and (e) the RMS
        regularizer. The value is a tuple containing the (a) x-values and (b)
        y-values during the optimization, and (c) the loss value. (The x- and
        y-value are only non-empty for 2D experiments.)
    '''
    for algo in algos:
        for _ in range(n):
            mu = max(0, np.random.uniform(0, 2) - 1)
            rate = np.exp(np.random.uniform(-8, -1))
            half = int(np.exp(np.random.uniform(0, 4)))
            reg = np.exp(np.random.uniform(-12, 0))
            yield (algo, rate, mu, half, reg), build_and_trace(
                algo, init, limit, momentum=mu, learning_rate=rate,
                rms_halflife=half, rms_regularizer=reg)
