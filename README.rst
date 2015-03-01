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

- Wrappers for several algorithms from ``scipy.optimize.minimize``.

The source code for ``theanopt`` lives at http://github.com/lmjohns3/theanopt,
the documentation lives at http://theanopt.readthedocs.org, and announcements
and discussion happen on the `mailing list`_.

.. _Theano: http://deeplearning.net/software/theano/
.. _mailing list: https://groups.google.com/forum/#!forum/theanopt

Example Code
============

This is pretty straightforward to model using Theano. Once you have an
expression for the loss, you can optimize it using ``theanopt``::

  import climate
  import theano
  import theano.tensor as TT
  import theanopt
  import my_data_set

  climate.enable_default_logging()

  A, B, K = 100, 1000, 10

  x = TT.matrix('x')

  u = theano.shared(np.random.randn(A, K).astype('f'), name='u')
  v = theano.shared(np.random.randn(K, B).astype('f'), name='v')

  err = TT.sqr(x - TT.dot(u, v))

  theanopt.minimize(
      loss=err.mean() + abs(u).mean() + (v * v).mean(),
      params=[u, v],
      inputs=[x],
      train=my_data_set.training,
      valid=my_data_set.validation,
      batch_size=A,
      monitors=(
          ('u<0.1', 100 * (abs(u) < 0.1).mean()),
          ('v<0.1', 100 * (abs(v) < 0.1).mean()),
      ),
  )
