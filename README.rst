The ``theanopt`` package provides tools for minimizing scalar loss functions. It
uses Python for rapid development, and under the hood Theano_ provides graph
optimization and fast computations on the GPU.

Several optimization algorithms are included:

- First-order stochastic gradient descent: :class:`SGD
  <theanopt.first_order.SGD>` and :class:`NAG <theanopt.first_order.NAG>`.

- Second-order stochastic gradient descent: :class:`HF
  <theanopt.second_order.HF>`.

- First-order stochastic techniques with adaptive learning rates: :class:`Rprop
  <theanopt.adaptive.Rprop>`, :class:`RmsProp <theanopt.adaptive.RmsProp>`,
  :class:`Equilibrated SGD <theanopt.adaptive.ESGD>`, and :class:`ADADELTA
  <theanopt.adaptive.ADADELTA>`.

- Several algorithms from ``scipy.optimize.minimize``.

The source code for ``theanopt`` lives at http://github.com/lmjohns3/theanopt,
the documentation lives at http://theanopt.readthedocs.org, and announcements
and discussion happen on the `mailing list`_.

.. _Theano: http://deeplearning.net/software/theano/
.. _mailing list: https://groups.google.com/forum/#!forum/theanopt

Example Code
============

Let's say you want to compute a low-rank approximation for a matrix ``x``. You
first set up a loss using Theano, and then optimize it using ``theanopt``::

  import climate
  import theano
  import theano.tensor as TT
  import theanopt
  import my_data_set

  climate.enable_default_logging()

  A, B, K = 1000, 2000, 10

  x = TT.matrix('x')

  u = theano.shared(np.random.randn(A, K).astype('f'), name='u')
  v = theano.shared(np.random.randn(K, B).astype('f'), name='v')

  err = TT.sqr(x - TT.dot(u, v))

  exp = theanopt.minimize(
      loss=err.mean() + abs(u).mean() + (v * v).mean(),
      params=[u, v],
      inputs=[x],
  )
