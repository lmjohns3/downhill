============
``DOWNHILL``
============

.. figure:: _static/rosenbrock-nag.png
   :figclass: banana

The ``downhill`` package provides algorithms for minimizing scalar loss
functions that are defined using Theano_.

Several optimization algorithms are included:

- First-order stochastic gradient descent: :class:`SGD
  <downhill.first_order.SGD>` and :class:`NAG <downhill.first_order.NAG>`.
- First-order stochastic techniques with adaptive learning rates: :class:`RProp
  <downhill.adaptive.RProp>`, :class:`RMSProp <downhill.adaptive.RMSProp>`,
  :class:`Equilibrated SGD <downhill.adaptive.ESGD>`, :class:`Adam
  <downhill.adaptive.Adam>`, and :class:`ADADELTA <downhill.adaptive.ADADELTA>`.

Most algorithms permit the use of momentum to accelerate progress.

The source code for ``downhill`` lives at http://github.com/lmjohns3/downhill,
the documentation lives at http://downhill.readthedocs.org, and announcements
and discussion happen on the `mailing list`_.

.. _Theano: http://deeplearning.net/software/theano/
.. _mailing list: https://groups.google.com/forum/#!forum/downhill-users

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

This is pretty straightforward to model using Theano. Once you set up the
appropriate variables and an expression for the loss, you can optimize the loss
with respect to the variables using ``downhill``::

  import climate
  import downhill
  import numpy as np
  import theano
  import theano.tensor as TT

  climate.enable_default_logging()

  A, B, K = 100, 1000, 10

  x = TT.matrix('x')

  y = np.arange(A * B).reshape((A, B)).astype('f')
  u = theano.shared(np.random.randn(A, K).astype('f'), name='u')
  v = theano.shared(np.random.randn(K, B).astype('f'), name='v')

  err = TT.sqr(x - TT.dot(u, v))

  downhill.minimize(
      loss=err.mean() + abs(u).mean() + (v * v).mean(),
      params=[u, v],
      inputs=[x],
      train=[y],
      batch_size=A,
      monitors=(
          ('u<0.1', 100 * (abs(u) < 0.1).mean()),
          ('v<0.1', 100 * (abs(v) < 0.1).mean()),
      ))

After optimization, you can get the :math:`u` and :math:`v` matrix values out of
the shared variables using ``u.get_value()`` and ``v.get_value()``.

Documentation
=============

.. toctree::
   :maxdepth: 2

   guide
   reference
