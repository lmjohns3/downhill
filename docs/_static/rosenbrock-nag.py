import climate
import downhill
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import theano

climate.enable_default_logging()

x = theano.shared(np.array([-1, 0], 'f'), name='x')

opt = downhill.build(
    'nag',
    loss=(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum(),
    params=[x],
    inputs=[],
    monitors=[('x', x[:-1].sum()), ('y', x[1:].sum())],
    monitor_gradients=True)

xs, ys, loss = [], [], []
for tm, _ in opt.iterate([[]],
                         learning_rate=0.001,
                         momentum=0.95,
                         max_gradient_norm=100):
    xs.append(tm['x'])
    ys.append(tm['y'])
    loss.append(tm['loss'])
    if len(loss) == 300:
        break

ax = plt.axes(projection='3d')

c = '#d62728'
ax.plot(xs, ys, zs=loss, linestyle='-',
        marker='o', color=c, mec=c, mfc='none',
        lw=3, mew=0.5, markersize=7, alpha=0.7)

X, Y = np.meshgrid(np.linspace(-1.1, 1.1, 127), np.linspace(-0.5, 1.7, 127))
Z = 100 * (Y - X ** 2) ** 2 + (1 - X) ** 2
ax.plot_surface(X, Y, Z, cmap='YlGnBu', lw=0, rstride=4, cstride=4, alpha=0.9)
ax.plot_wireframe(X, Y, Z, lw=0.5, rstride=4, cstride=4, color='#333333', alpha=0.7)
ax.plot([1], [1], zs=[1], marker='x', mew=3, markersize=10, color='#111111')

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-0.5, 1.7)
ax.view_init(azim=10, elev=45)

ax.w_xaxis.set_pane_color((1, 1, 1, 1))
ax.w_yaxis.set_pane_color((1, 1, 1, 1))
ax.w_zaxis.set_pane_color((1, 1, 1, 1))

plt.savefig('rosenbrock-nag.png')
plt.show()
