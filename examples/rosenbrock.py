import climate
import downhill
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import mpl_toolkits.mplot3d.axes3d
import numpy as np
import theano
import theano.tensor as TT

climate.enable_default_logging()


_, ax = plt.subplots(1, 1)

# run several optimizers for comparison.
for i, (algo, label, kw) in enumerate((
        ('sgd', 'SGD - Momentum 0', {}),
        ('sgd', 'SGD - Momentum 0.5', dict(momentum=0.5, nesterov=False)),
        ('rmsprop', 'RMSProp - Momentum 0', {}),
        ('rmsprop', 'RMSProp - Momentum 0.5', dict(momentum=0.5, nesterov=False)),
        ('adam', 'Adam - Momentum 0', {}),
        #('esgd', 'ESGD - Momentum 0', {}),
        ('rprop', 'RProp - Momentum 0', {}),
        ('adadelta', 'ADADELTA - Momentum 0', {}),
        )):
    print(label)
    x = theano.shared(np.array([-1.1, -0.4], 'f'), name='x')
    opt = downhill.build(
        algo,
        loss=(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum(),
        params=[x],
        inputs=[],
        monitors=[('x', x[:-1].sum()), ('y', x[1:].sum())])
    xs, ys = [], []
    for tm, _ in opt.iteropt([[]],
                             max_gradient_clip=1,
                             min_improvement=0,
                             learning_rate=0.01,
                             patience=0,
                             **kw):
        xs.append(tm['x'])
        ys.append(tm['y'])
        if len(xs) == 100:
            break
    ax.plot(np.array(xs), np.array(ys) + 0.05 * i,
            'o-', label=label, alpha=0.3)

# make a contour plot of the rosenbrock function surface.
a = b = np.arange(-1.2, 1.2, 0.05)
X, Y = np.meshgrid(a, b)
Z = 100 * (Y - X ** 2) ** 2 + (1 - X) ** 2

ax.plot([1], [1], 'x', mew=2, color='#111111')
ax.contourf(X, Y, Z, np.logspace(0, 3, 10))

plt.legend(loc='lower right')
plt.show()
