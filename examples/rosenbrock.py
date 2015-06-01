# -*- coding: utf-8 -*-

'''Optimization example using the two-dimensional Rosenbrock "banana" function.

This example trains up several optimization algorithms and displays the
performance of each algorithm across several different (randomly-chosen)
hyperparameter settings.

This example is meant to show how different optimization algorithms perform when
given the same optimization problem. Many of the algorithms' performances are
strongly dependent on the values of various hyperparameters, such as the
learning rate and momentum values.
'''

import climate
import downhill
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as TT

climate.enable_default_logging()

COLORS = ('#d62728 #1f77b4 #2ca02c #9467bd #ff7f0e '
          '#e377c2 #8c564b #bcbd22 #7f7f7f #17becf').split()


def build(algo):
    '''Build and return an optimizer for the rosenbrock function.

    In downhill, an optimizer can be constructed using the build() top-level
    function. This function requires several Theano quantities such as the loss
    being optimized and the parameters to update during optimization.
    '''
    x = theano.shared(np.array([-1.1, 0], 'f'), name='x')
    return downhill.build(
        algo,
        loss=(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum(),
        params=[x],
        inputs=[],
        # this gives us access to the x and y locations during optimization.
        monitors=[('x', x[:-1].sum()), ('y', x[1:].sum())],
        monitor_gradients=True)


# Default hyperparameters for some learning algorithms. It would be fun to
# include these in the hyperparameter search but for now we'll just keep them
# fixed.
DEFAULTS = dict(
    ADADELTA=dict(rms_halflife=7, rms_regularizer=1e-2, max_gradient_clip=1e10),
    ESGD=dict(rms_halflife=50, rms_regularizer=1e-2, max_gradient_clip=1e10),
)

def build_and_trace(algo, limit=100, **kwargs):
    '''Run an optimizer on the rosenbrock function. Return xs, ys, and losses.

    In downhill, optimization algorithms can be iterated over to progressively
    minimize the loss. At each iteration, the optimizer yields a dictionary of
    monitor values that were computed during that iteration. Here we build an
    optimizer and then run it for a fixed number of iterations.
    '''
    kw = dict(
        max_gradient_clip=1,
        min_improvement=0,
        patience=100,
    )
    kw.update(DEFAULTS.get(algo, {}))
    kw.update(kwargs)
    xs, ys, loss = [], [], []
    for tm, _ in build(algo).iteropt([[]], **kw):
        xs.append(tm['x'])
        ys.append(tm['y'])
        loss.append(tm['loss'])
        if len(xs) == limit:
            break
    return xs, ys, loss


def plot(results):
    '''Plot a set of results on top of a contour of the banana function.

    The results should be a sequence of (label, xs, ys) -- one of these tuples
    for each optimizer that we want to plot.
    '''
    _, ax = plt.subplots(1, 1)

    for color, (label, xs, ys) in zip(COLORS, results):
        ax.plot(xs, ys, 'o-', color=color, label=label,
                alpha=0.8, lw=2, markersize=5,
                mew=1, mec=color, mfc='none')

    # make a contour plot of the rosenbrock function surface.
    X, Y = np.meshgrid(np.linspace(-1.2, 1.2, 31), np.linspace(-0.7, 1.7, 31))
    Z = 100 * (Y - X ** 2) ** 2 + (1 - X) ** 2
    ax.plot([1], [1], 'x', mew=3, markersize=10, color='#111111')
    ax.contourf(X, Y, Z, np.logspace(-1, 2.7, 31), cmap='gray_r')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.7, 1.7)

    plt.legend(loc='lower right')
    plt.show()


def min_loss(item):
    '''A helper for sorting optimization results by loss.'''
    label, (xs, ys, loss) = item
    return min(loss)


# Here we run several optimizers for comparison. Each optimizer is run a fixed
# number of times with random hyperparameter values, and the results are stored
# in a dictionary. Below, we sort these and plot the best ones.

results = {}
for algo in 'SGD NAG RMSProp RProp Adam ADADELTA ESGD'.split():
    for _ in range(5):
        mu = np.random.choice([0, 0.1, 0.5, 0.9])
        rate = np.random.choice([1e-1, 1e-2, 1e-3, 1e-4])
        label = '{} µ={} r={}'.format(algo, mu, rate)
        results[label] = build_and_trace(algo, momentum=mu, learning_rate=rate)
    # Uncomment below to plot results for just this optimization algorithm.
    #plot((label, xs, ys) for label, (xs, ys, _)
    #     in sorted(results.items(), key=min_loss)
    #     if label.startswith(algo))

plot(('({:5f}) {}'.format(min(loss), label), xs, ys)
     for label, (xs, ys, loss)
     in sorted(results.items(), key=min_loss))
