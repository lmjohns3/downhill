.. figure:: _static/rosenbrock-nag.png
   :figclass: banana

The ``downhill`` package provides algorithms for minimizing scalar loss
functions that are defined using Theano_.

Several optimization algorithms are included:

- :class:`ADADELTA <downhill.adaptive.ADADELTA>`
- :class:`ADAGRAD <downhill.adaptive.ADAGRAD>`
- :class:`Adam <downhill.adaptive.Adam>`
- :class:`Equilibrated SGD <downhill.adaptive.ESGD>`
- :class:`Nesterov's Accelerated Gradient <downhill.first_order.NAG>`
- :class:`RMSProp <downhill.adaptive.RMSProp>`
- :class:`Resilient Backpropagation <downhill.adaptive.RProp>`
- :class:`Stochastic Gradient Descent <downhill.first_order.SGD>`

All algorithms permit the use of regular or Nesterov-style momentum as well.

The source code for ``downhill`` lives at http://github.com/lmjohns3/downhill,
the documentation lives at http://downhill.readthedocs.org, and announcements
and discussion happen on the `mailing list`_.

.. _Theano: http://deeplearning.net/software/theano/
.. _mailing list: https://groups.google.com/forum/#!forum/downhill-users

Quick Start: Matrix Factorization
=================================

Let's say you want to compute a sparse, low-rank approximation for some
1000-dimensional data that you have lying around. You can represent a batch of
:math:`m` of data points :math:`X \in \mathbb{R}^{m \times 1000}` as the product
of a sparse coefficient matrix :math:`U \in \mathbb{R}^{m \times k}` and a
low-rank basis matrix :math:`V \in \mathbb{R}^{k \times 1000}`. You might
represent the loss as

.. math::

   \mathcal{L} = \| X - UV \|_2^2 + \alpha \| U \|_1 + \beta \| V \|_2

where the first term represents the approximation error, the second represents
the sparsity of the representation, and the third prevents the basis vectors
from growing too large.

This is pretty straightforward to model using Theano. Once you set up the
appropriate variables and an expression for the loss, you can optimize the loss
with respect to the variables using a single call to :func:`downhill.minimize`::

  import downhill
  import numpy as np
  import theano
  import theano.tensor as TT

  FLOAT = 'df'[theano.config.floatX == 'float32']

  def rand(a, b):
      return np.random.randn(a, b).astype(FLOAT)

  A, B, K = 20, 5, 3

  # Set up a matrix factorization problem to optimize.
  u = theano.shared(rand(A, K), name='u')
  v = theano.shared(rand(K, B), name='v')
  z = TT.matrix()
  err = TT.sqr(z - TT.dot(u, v))
  loss = err.mean() + abs(u).mean() + (v * v).mean()

  # Minimize the regularized loss with respect to a data matrix.
  y = np.dot(rand(A, K), rand(K, B)) + rand(A, B)

  # Monitor during optimization.
  monitors = (('err', err.mean()),
              ('|u|<0.1', (abs(u) < 0.1).mean()),
              ('|v|<0.1', (abs(v) < 0.1).mean()))

  downhill.minimize(
      loss=loss,
      train=[y],
      patience=0,
      batch_size=A,                 # Process y as a single batch.
      max_gradient_norm=1,          # Prevent gradient explosion!
      learning_rate=0.1,
      monitors=monitors,
      monitor_gradients=True)

  # Print out the optimized coefficients u and basis v.
  print('u =', u.get_value())
  print('v =', v.get_value())

If you prefer to maintain more control over your model during optimization,
downhill provides an iterative optimization interface::

  opt = downhill.build(algo='rmsprop',
                       loss=loss,
                       monitors=monitors,
                       monitor_gradients=True)

  for metrics, _ in opt.iterate(train=[[y]],
                                patience=0,
                                batch_size=A,
                                max_gradient_norm=1,
                                learning_rate=0.1):
      print(metrics)

If that's still not enough, you can just plain ask downhill for the updates to
your model variables and do everything else yourself::

  updates = downhill.build('rmsprop', loss).get_updates(
      batch_size=A, max_gradient_norm=1, learning_rate=0.1)
  func = theano.function([z], loss, updates=list(updates))
  for _ in range(100):
      print(func(y))  # Evaluate func and apply variable updates.

Documentation
=============

.. toctree::
   :maxdepth: 2

   guide
   reference

Indices and tables
==================

- :ref:`genindex`
- :ref:`modindex`
