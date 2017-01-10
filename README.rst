.. image:: https://travis-ci.org/lmjohns3/downhill.svg
.. image:: https://coveralls.io/repos/lmjohns3/downhill/badge.svg
   :target: https://coveralls.io/r/lmjohns3/downhill
.. image:: http://depsy.org/api/package/pypi/downhill/badge.svg
   :target: http://depsy.org/package/python/downhill

============
``DOWNHILL``
============

The ``downhill`` package provides algorithms for minimizing scalar loss
functions that are defined using Theano_.

Several optimization algorithms are included:

- ADADELTA_
- ADAGRAD_
- Adam_
- `Equilibrated SGD`_
- `Nesterov's Accelerated Gradient`_
- RMSProp_
- `Resilient Backpropagation`_
- `Stochastic Gradient Descent`_

All algorithms permit the use of regular or Nesterov-style momentum as well.

.. _Theano: http://deeplearning.net/software/theano/

.. _Stochastic Gradient Descent: http://downhill.readthedocs.org/en/stable/generated/downhill.first_order.SGD.html
.. _Nesterov's Accelerated Gradient: http://downhill.readthedocs.org/en/stable/generated/downhill.first_order.NAG.html
.. _Resilient Backpropagation: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.RProp.html
.. _ADAGRAD: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.ADAGRAD.html
.. _RMSProp: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.RMSProp.html
.. _ADADELTA: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.ADADELTA.html
.. _Adam: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.Adam.html
.. _Equilibrated SGD: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.ESGD.html

Quick Start: Matrix Factorization
=================================

Let's say you have 100 samples of 1000-dimensional data, and you want to
represent your data as 100 coefficients in a 10-dimensional basis. This is
pretty straightforward to model using Theano: you can use a matrix
multiplication as the data model, a squared-error term for optimization, and a
sparse regularizer to encourage small coefficient values.

Once you have constructed an expression for the loss, you can optimize it with a
single call to ``downhill.minimize``:

.. code:: python

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
downhill provides an iterative optimization interface:

.. code:: python

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
your model variables and do everything else yourself:

.. code:: python

  updates = downhill.build('rmsprop', loss).get_updates(
      batch_size=A, max_gradient_norm=1, learning_rate=0.1)
  func = theano.function([z], loss, updates=list(updates))
  for _ in range(100):
      print(func(y))  # Evaluate func and apply variable updates.

More Information
================

Source: http://github.com/lmjohns3/downhill

Documentation: http://downhill.readthedocs.org

Mailing list: https://groups.google.com/forum/#!forum/downhill-users
