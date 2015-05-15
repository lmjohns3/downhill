The ``downhill`` package provides tools for minimizing scalar loss functions. It
uses Python for rapid development, and under the hood Theano_ provides graph
optimization and fast computations on the GPU.

Several optimization algorithms are included:

- First-order stochastic gradient descent: SGD_ and NAG_.
- Second-order stochastic gradient descent: `Hessian-free`_.
- First-order stochastic techniques with adaptive learning rates: RProp_,
  RMSProp_, `Equilibrated SGD`_, and ADADELTA_.
- Wrappers for several algorithms from ``scipy.optimize.minimize``.

.. _Theano: http://deeplearning.net/software/theano/

.. _SGD: http://downhill.readthedocs.org/en/stable/generated/downhill.first_order.SGD.html
.. _NAG: http://downhill.readthedocs.org/en/stable/generated/downhill.first_order.NAG.html
.. _Hessian-free: http://downhill.readthedocs.org/en/stable/generated/downhill.second_order.HF.html
.. _RProp: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.RProp.html
.. _RMSProp: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.RMSProp.html
.. _ADADELTA: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.ADADELTA.html
.. _Equilibrated SGD: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.ESGD.html

Example Code
============

Let's say you have 100 samples of 1000-dimensional data, and you want to
represent your data as 100 coefficients in a 10-dimensional basis. This is
pretty straightforward to model using Theano, using a matrix multiplication.
Once you have constructed an expression for the loss, you can optimize it using
``downhill``::

  import climate
  import theano
  import theano.tensor as TT
  import downhill
  import my_data_set

  climate.enable_default_logging()

  A, B, K = 100, 1000, 10

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

More Information
================

Source: http://github.com/lmjohns3/downhill

Documentation: http://downhill.readthedocs.org

Mailing list: https://groups.google.com/forum/#!forum/downhill-users
