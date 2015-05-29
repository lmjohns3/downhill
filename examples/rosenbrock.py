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
for i, (algo, label, m, n, kw) in enumerate((
        ('sgd', 'SGD - Momentum 0', 0, False, {}),
        ('sgd', 'SGD - Momentum 0.1', 0.1, False, {}),
        ('sgd', 'SGD - Momentum 0.5', 0.5, False, {}),
        ('sgd', 'SGD - Nesterov 0.5', 0.5, True, {}),
        ('rmsprop', 'RMSProp - Momentum 0', 0, False, {}),
        ('rmsprop', 'RMSProp - Momentum 0.5', 0.5, False, {}),
        ('adam', 'Adam - Momentum 0', 0, False, {}),
        ('adam', 'Adam - Momentum 0.5', 0.5, False, {}),
        #('esgd', 'ESGD - Momentum 0', dict(learning_rate=0.02)),
        ('rprop', 'RProp - Momentum 0', 0, False, {}),
        ('rprop', 'RProp - Momentum 0.5', 0.5, False, {}),
        ('adadelta', 'ADADELTA - Momentum 0', 0, False,
         dict(rms_halflife=10, rms_regularizer=1e-2)),
        ('adadelta', 'ADADELTA - Momentum 0.5', 0.5, False,
         dict(rms_halflife=10, rms_regularizer=1e-2)),
        )):
    print(label)
    x = theano.shared(np.array([-1.1, -0.4 + 0.03 * i], 'f'), name='x')
    opt = downhill.build(
        algo,
        loss=(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum(),
        params=[x],
        inputs=[],
        monitors=[('x', x[:-1].sum()), ('y', x[1:].sum())],
        monitor_gradients=True)
    xs, ys = [], []
    for tm, _ in opt.iteropt([[]],
                             learning_rate=0.02,
                             momentum=m,
                             nesterov=n,
                             max_gradient_clip=10,
                             min_improvement=0,
                             patience=0,
                             **kw):
        xs.append(tm['x'])
        ys.append(tm['y'])
        if len(xs) == 100:
            break
    ax.plot(xs, ys, 'o-', label=label, alpha=0.3)

# make a contour plot of the rosenbrock function surface.
a = b = np.arange(-1.2, 1.2, 0.05)
X, Y = np.meshgrid(a, b)
Z = 100 * (Y - X ** 2) ** 2 + (1 - X) ** 2

ax.plot([1], [1], 'x', mew=2, color='#111111')
ax.contourf(X, Y, Z, np.logspace(0, 3, 10))

plt.legend(loc='lower right')
plt.show()
