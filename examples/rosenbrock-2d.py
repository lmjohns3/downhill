'''Optimization example using the two-dimensional Rosenbrock "banana" function.

This example trains up several optimization algorithms and displays the
performance of each algorithm across several different (randomly-chosen)
hyperparameter settings.

This example is meant to show how different optimization algorithms perform when
given the same optimization problem. Many of the algorithms' performances are
strongly dependent on the values of various hyperparameters, such as the
learning rate and momentum values.
'''

import matplotlib.pyplot as plt
import numpy as np

import rosenbrock


def by_loss(item):
    '''Helper for sorting optimization runs by their final loss value.'''
    label, (xs, ys, loss) = item
    return loss


def make_label(loss, key):
    '''Create a legend label for an optimization run.'''
    algo, rate, mu, half, reg = key
    slots, args = ['{:.3f}', '{}', 'm={:.3f}'], [loss, algo, mu]
    if algo in 'SGD NAG RMSProp Adam ESGD'.split():
        slots.append('lr={:.2e}')
        args.append(rate)
    if algo in 'RMSProp ADADELTA ESGD'.split():
        slots.append('rmsh={}')
        args.append(half)
        slots.append('rmsr={:.2e}')
        args.append(reg)
    return ' '.join(slots).format(*args)


# Here we run a number of rosenbrock optimization algorithms and measure their
# performance. Below we plot the results.

algos = 'SGD NAG RMSProp RProp Adam ADADELTA ESGD'.split()
results = ((make_label(loss, key), xs, ys)
           for key, (xs, ys, loss)
           in sorted(rosenbrock.test(algos), key=by_loss))

_, ax = plt.subplots(1, 1)

for color, (label, xs, ys) in zip(rosenbrock.COLORS, results):
    ax.plot(xs, ys, 'o-', color=color, label=label,
            alpha=0.8, lw=2, markersize=5,
            mew=1, mec=color, mfc='none')

# make a contour plot of the rosenbrock function surface.
X, Y = np.meshgrid(np.linspace(-1.3, 1.3, 31), np.linspace(-0.9, 1.7, 31))
Z = 100 * (Y - X ** 2) ** 2 + (1 - X) ** 2
ax.plot([1], [1], 'x', mew=3, markersize=10, color='#111111')
ax.contourf(X, Y, Z, np.logspace(-1, 3, 31), cmap='gray_r')

ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-0.9, 1.7)

plt.legend(loc='lower right')
plt.show()
