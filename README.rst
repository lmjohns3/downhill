============
``DOWNHILL``
============

.. image:: https://travis-ci.org/lmjohns3/downhill.svg
.. image:: https://coveralls.io/repos/lmjohns3/downhill/badge.svg
   :target: https://coveralls.io/r/lmjohns3/downhill

The ``downhill`` package provides algorithms for minimizing scalar loss
functions that are defined using Theano_.

Several optimization algorithms are included:

- First-order stochastic gradient descent: SGD_ and NAG_.
- First-order stochastic techniques with adaptive learning rates: RProp_,
  RMSProp_, `Equilibrated SGD`_, Adam_, and ADADELTA_.
- Wrappers for several algorithms from ``scipy.optimize.minimize``.

.. _Theano: http://deeplearning.net/software/theano/

.. _SGD: http://downhill.readthedocs.org/en/stable/generated/downhill.first_order.SGD.html
.. _NAG: http://downhill.readthedocs.org/en/stable/generated/downhill.first_order.NAG.html
.. _RProp: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.RProp.html
.. _RMSProp: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.RMSProp.html
.. _ADADELTA: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.ADADELTA.html
.. _Adam: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.Adam.html
.. _Equilibrated SGD: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.ESGD.html

Example Code
============

Let's say you have 100 samples of 1000-dimensional data, and you want to
represent your data as 100 coefficients in a 10-dimensional basis. This is
pretty straightforward to model using Theano, using a matrix multiplication.
Once you have constructed an expression for the loss, you can optimize it using
``downhill``::

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

More Information
================

Source: http://github.com/lmjohns3/downhill

Documentation: http://downhill.readthedocs.org

Mailing list: https://groups.google.com/forum/#!forum/downhill-users
