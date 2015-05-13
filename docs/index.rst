============
``DOWNHILL``
============

The ``downhill`` package provides tools for minimizing scalar loss functions. It
uses Python for rapid development, and under the hood Theano_ provides graph
optimization and fast computations on the GPU.

Several optimization algorithms are included:

- First-order stochastic gradient descent: :class:`SGD
  <downhill.first_order.SGD>` and :class:`NAG <downhill.first_order.NAG>`.
- First-order stochastic techniques with adaptive learning rates: :class:`RProp
  <downhill.adaptive.RProp>`, :class:`RMSProp <downhill.adaptive.RMSProp>`,
  :class:`Equilibrated SGD <downhill.adaptive.ESGD>`, and :class:`ADADELTA
  <downhill.adaptive.ADADELTA>`.
- Wrappers for several algorithms from ``scipy.optimize.minimize``.

The source code for ``downhill`` lives at http://github.com/lmjohns3/downhill,
the documentation lives at http://downhill.readthedocs.org, and announcements
and discussion happen on the `mailing list`_.

.. _Theano: http://deeplearning.net/software/theano/
.. _mailing list: https://groups.google.com/forum/#!forum/downhill

Example Code
============

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

This is pretty straightforward to model using Theano. Once you have set up the
appropriate variables and an expression for the loss, you can optimize it using
``downhill``::

  import climate
  import theano
  import theano.tensor as TT
  import downhill
  import my_data_set

  climate.enable_default_logging()

  A, B, K = 1000, 2000, 10

  x = TT.matrix('x')

  u = theano.shared(np.random.randn(A, K).astype('f'), name='u')
  v = theano.shared(np.random.randn(K, B).astype('f'), name='v')

  err = TT.sqr(x - TT.dot(u, v))

  downhill.minimize(
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

After optimization, you can get the :math:`u` and :math:`v` matrix values out of
the shared variables using ``get_value()``.

Documentation
=============

.. toctree::
   :maxdepth: 2

   quickstart
   reference
