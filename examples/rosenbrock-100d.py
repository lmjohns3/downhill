'''Optimization example using the 100-dimensional Rosenbrock "banana" function.

This example trains up several optimization algorithms with randomly chosen
hyperparameters and shows four histograms of the performance spectrum of each
hyperparameter.

This example is meant to show how optimization hyperparameters affect
performance across different optimization algorithms.

Due to the large number of optimizers that are evaluated in this example, it can
take a good while to run.
'''

import itertools
import matplotlib.pyplot as plt
import numpy as np

import rosenbrock


algos = 'NAG RMSProp Adam ADADELTA'.split()
results = rosenbrock.test(algos, n=3, init=[-1] * 100, limit=1000)


# Here we make plots of the marginal performance of each of the four
# hyperparameters. These are intended to get a sense of how random
# hyperparameter selection gives a decent idea of how different algorithms
# perform.

_, ((rate_ax, mu_ax), (half_ax, reg_ax)) = plt.subplots(2, 2)

by_algo = itertools.groupby(sorted(results), lambda item: item[0][0])
for color, (algo, items) in zip(rosenbrock.COLORS, by_algo):
    items = list(items)
    values = np.zeros((len(items), 5), 'f')
    for i, ((_, rate, mu, half, reg), (_, _, loss)) in enumerate(items):
        values[i] = [rate, mu, half, reg, loss]
    rates, mus, halfs, regs, losses = values.T
    kw = dict(alpha=0.8, markersize=5, mew=2, mfc='none', mec=color)
    rate_ax.plot(rates, losses, 'o', label=algo, **kw)
    mu_ax.plot(mus, losses, 'o', label=algo, **kw)
    half_ax.plot(halfs, losses, 'o', label=algo, **kw)
    reg_ax.plot(regs, losses, 'o', label=algo, **kw)

for ax in [rate_ax, mu_ax, half_ax, reg_ax]:
    ax.set_yscale('log')
    ax.set_ylim(None, 4e4)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('outward', 3))
    ax.spines['left'].set_position(('outward', 3))
    if ax != mu_ax:
        ax.set_xscale('log')

rate_ax.set_ylabel('Loss')
rate_ax.set_xlabel('Rate')

mu_ax.set_xlabel('Momentum')
mu_ax.set_xlim(-0.05, 1.05)

half_ax.set_ylabel('Loss')
half_ax.set_xlabel('RMS Halflife')

reg_ax.set_xlabel('RMS Regularizer')

plt.legend()
plt.show()
