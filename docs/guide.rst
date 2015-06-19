==========
User Guide
==========

You are probably reading this guide because you have a problem.

There are many problems in the world, and many ways of thinking about solving
them. Happily, some---many---problems can be described mathematically using a
"loss" function, which takes a potential solution for your problem and returns a
single number indicating how terrible that solution is.

If you can express your problem using a loss function, then it's possible---even
likely---that you can then use a computer to solve your problem for you. This is
what ``downhill`` does: given a computational formulation of a loss for a
problem, the optimization routines in ``downhill`` can compute a series of
ever-better solutions to your problem.

This guide describes how that works.

.. _creating-loss:

Creating a Loss
===============

Many types of problems can be formulated in terms of a scalar `"loss" function`_
that ought to be minimized. The "loss" for a problem:

- is computed with respect to a potential solution to a problem, and
- is a scalar quantity---just a single number.

A few examples of problems and their associated losses might include:

- Categorizing pictures into "elephants" versus "acrobats"; the loss might be
  the number of mistakes that are made on a given set of test pictures.
- Allocating funds to provide a given set of public services; the loss might be
  the monetary cost of the budget.
- Computing the actions of a robot to achieve a goal; the loss might be the
  total energy consumed.

This guide will use linear regression as a running example. Suppose you've made
some measurements of, say, the sizes and prices of various houses for sale where
you live. You want to describe the relationship between the size (let's
represent it as :math:`x_i`) and the price (:math:`y_i`) by fitting a line to
the measurements you've made.

So you need to take the data points that you collected and somehow use them to
compute a slope :math:`m` and an intercept :math:`b` such that the resulting
line :math:`y_i = m x_i + b` passes as closely as possible to your data points.
In this example, the loss :math:`\mathcal{L}` might be expressed as the sum of
the differences between the values on the line and the observed data:

.. math::
   \mathcal{L}(m,b) = \sum_{i=1}^N \| m x_i + b - y_i \|_2^2

.. _"loss" function: https://en.wikipedia.org/wiki/Loss_function

Using Theano
------------

Well, you've formulated a loss for this regression problem. Now it's time to use
``downhill`` to minimize it, right?

Not so fast ... the ``downhill`` package provides routines for optimizing scalar
loss functions, but there's a catch: the loss functions must be defined using
Theano_, a Python framework for describing computation graphs. It takes a bit of
getting used to, but we'll walk through the example here.

In Theano, you need to define `shared variables`_ for each of the parameters in
your model, and `symbolic inputs`_ for the data that you'll use to evaluate your
loss. We'll start with the shared variables::

  import numpy as np
  import theano
  import theano.tensor as TT

  m = theano.shared(np.ones((1, ), 'f'), name='m')
  b = theano.shared(np.zeros((1, ), 'f'), name='b')

This sets up a vector with one 1 for :math:`m`, and a vector with one 0 for
:math:`b`. The values contained inside these shared variables will be adjusted
automatically by the optimization algorithms in ``downhill``.

Next, you need to define some symbols that represent the data needed to compute
the loss::

  x = TT.vector('x')
  y = TT.vector('y')

These symbolic vectors represent the inputs :math:`[x_1 \dots x_N]` and
:math:`[y_1 \dots y_N]` to compute the loss. Finally, you can define the loss
itself::

  loss = TT.sqr(m * x + b - y).sum()

This tells Theano to multiply the data vector ``x`` by the value stored in the
shared ``m`` variable, add the value stored in the shared ``b`` variable, and
then subtract the data vector ``y``. Square that vector elementwise, and then
add up all of the components of the result to get the loss.

Note that none of these operations have actually been computed; instead, you've
instructed Theano *how* to compute the loss, if you give it some values for
``x`` and ``y``. This is the bizarre thing about Theano: it looks like you're
computing things, but you're actually just telling the computer how to compute
things in the future.

.. _Theano: http://deeplearning.net/software/theano/
.. _shared variables: http://deeplearning.net/software/theano/tutorial/examples.html#using-shared-variables
.. _symbolic inputs: http://deeplearning.net/software/theano/tutorial/adding.html

.. _minimizing-loss:

Minimizing a Loss
=================

The ``downhill`` package provides a single high-level function,
:func:`downhill.minimize`, that can be used as a black-box optimizer for losses.
In addition, there are lower-level calls that provide more control over the
interaction between your code and ``downhill``.

.. _creating-optimizer:

Creating an Optimizer
=====================

.. _providing-data:

Providing Data
==============

You might have noticed that the formulation of the loss given above contains a
sum over all of the observed data points :math:`(x_i, y_i)`. This is a very
common state of affairs for many types of losses.

For most problems it's not possible to collect all the possible data points out
there! So you'll never actually know the "real" value of the loss for your
problem; instead you have to estimate it by collecting some data and hoping that
your collection is somehow representative of the data you'll encounter in the
future.

Either way, you'll often need to provide data to ``downhill`` so that you can
compute the loss and optimize the parameters. There are two ways of passing data
to ``downhill``: using arrays and using callables.

.. _data-training-validation:

Training and Validation
-----------------------

.. _data-using-arrays:

Using Arrays
------------

A fairly typical use case for optimizing a loss for a small-ish problem is to
construct a ``numpy`` array containing the data you have::

  dataset = np.load(filename)
  downhill.minimize(..., train=dataset)

Sometimes the data available for training a network model exceeds the available
resources (e.g., memory) on the computer at hand. There are several ways of
handling this type of situation. If your data are already in a ``numpy`` array
stored on disk, you might want to try loading the array using ``mmap``::

  dataset = np.load(filename, mmap_mode='r')
  downhill.minimize(..., train=dataset)

Alternatively, you might want to load just part of the data and train on that,
then load another part and train on it::

  for filename in filenames:
      dataset = np.load(filename, mmap_mode='r')
      downhill.minimize(..., train=dataset)

Finally, you can potentially handle large datasets by using a callable to
provide data to the training algorithm.

.. _data-using-callables:

Using Callables
---------------

Instead of an array of data, you can provide a callable for a dataset. This
callable must take no arguments and must return one or more ``numpy`` arrays of
the proper shape for your loss.

During minimization, the callable will be invoked every time the optimization
algorithm requires a batch of training (or validation) data. Therefore, your
callable should return at least one array containing a batch of data; if your
model requires multiple arrays per batch (e.g., if you are minimizing a loss
that requires some "input" data as well as some "output" data), then your
callable should return a list containing the correct number of arrays (e.g., an
array of "inputs" and the corresponding "outputs").

For example, this code defines a ``batch()`` helper that could be used for a
loss that needs one input. The callable chooses a random dataset and a random
offset for each batch::

  SOURCES = 'foo.npy', 'bar.npy', 'baz.npy'
  BATCH_SIZE = 64

  def batch():
      X = np.load(np.random.choice(SOURCES), mmap_mode='r')
      i = np.random.randint(len(X))
      return X[i:i+BATCH_SIZE]

  # ...

  exp.train(batch)

If you need to maintain more state than is reasonable from a single closure, you
can also encapsulate the callable inside a class. Just make sure instances of
the class are callable by defining the ``__call__`` method. For example, this
class loads data from a series of ``numpy`` arrays on disk, but only loads one
of the on-disk arrays into memory at a given time::

  class Loader:
      def __init__(sources=('foo.npy', 'bar.npy', 'baz.npy'), batch_size=64):
          self.sources = sources
          self.batch_size = batch_size
          self.src = -1
          self.idx = 0
          self.X = ()

      def __call__(self):
          if self.idx + self.batch_size > len(self.X):
              self.idx = 0
              self.src = (self.src + 1) % len(self.sources)
              self.X = np.load(self.sources[self.src], mmap_mode='r')
          try:
              return self.X[self.idx:self.idx+self.batch_size]
          finally:
              self.idx += self.batch_size

  # ...

  exp.train(Loader())

There are almost limitless possibilities for using callables to interface with
the optimization process.
