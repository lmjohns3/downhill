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
line :math:`y = m x + b` passes as closely as possible to your data points. In
this example, the loss :math:`\mathcal{L}` might be expressed as the sum of the
differences between the values on the line and the observed data:

.. math::
   \mathcal{L}(m,b) = \sum_{i=1}^N ( m x_i + b - y_i )^2

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

  import downhill
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
interaction between your code and ``downhill``. First, we'll look at the
high-level minimize function and then discuss some common optimization
hyperparameters.

Once you've defined your loss using Theano, you can minimize it with a single
function call. Here, we'll minimize the loss defined above::

  downhill.minimize(loss=loss, inputs=[x, y], train=[sizes, prices])

You just specify the loss to minimize, the inputs that the loss requires, and a
set of "training" data to use for computing the loss. The ``downhill`` code will
select an optimization algorithm, identify shared variables in the loss that
need optimization, and run the optimization process to completion. After the
minimization has finished, the shared variables in your loss will be updated to
their optimal values.

There is much to say about data---see :ref:`providing-data` for more
information---but briefly, the training data is typically a list of ``numpy``
arrays of the measurements you've made for your problem; for the house price
regression, the arrays for house size and house price might be set up like
this::

  sizes = np.array([1200, 2013, 8129, 2431, 2211])
  prices = np.array([103020, 203310, 3922013, 224321, 449020])

While running the optimization procedure, there are several different algorithms
to choose from, and there are also several common hyperparameters that might be
important to tune properly to get the best performance.

.. _algorithm:

Algorithms
----------

The following algorithms are currently available in ``downhill``:

- ``'adadelta'`` --- :class:`ADADELTA <downhill.adaptive.ADADELTA>`
- ``'adagrad'`` --- :class:`ADAGRAD <downhill.adaptive.ADAGRAD>`
- ``'adam'`` --- :class:`Adam <downhill.adaptive.Adam>`
- ``'esgd'`` --- :class:`Equilibrated SGD <downhill.adaptive.ESGD>`
- ``'nag'`` --- :class:`Nesterov's Accelerated Gradient <downhill.first_order.NAG>`
- ``'rmsprop'`` --- :class:`RMSProp <downhill.adaptive.RMSProp>`
- ``'rprop'`` --- :class:`Resilient Backpropagation <downhill.adaptive.RProp>`
- ``'sgd'`` --- :class:`Stochastic Gradient Descent <downhill.first_order.SGD>`

To select an algorithm, specify its name using the ``algo`` keyword argument::

  downhill.minimize(..., algo='adadelta')

Different algorithms have different performance characteristics, different
numbers of hyperparameters to tune, and might be better or worse for particular
problems. In general, several of the the adaptive procedures seem to work well
across different problems, particularly :class:`Adam <downhill.adaptive.Adam>`,
:class:`ADADELTA <downhill.adaptive.ADADELTA>`, and :class:`RMSProp
<downhill.adaptive.RMSProp>`.

.. _learning-rate:

Learning Rate
-------------

Most stochastic gradient optimization methods make small parameter updates based
on the local gradient of the loss at each step in the optimization procedure.
Intuitively, parameters in a model are updated by subtracting a small portion of
the local derivative from the current parameter value. Mathematically, this is
written as:

.. math::

   \theta_{t+1} = \theta_t - \alpha \left.
      \frac{\partial\mathcal{L}}{\partial\theta} \right|_{\theta_t}

where :math:`\mathcal{L}` is the loss function being optimized, :math:`\theta`
is the value of a parameter in the model (e.g., :math:`m` or :math:`b` for the
regression problem) at optimization step :math:`t`, :math:`\alpha` is the
learning rate, and :math:`\frac{\partial\mathcal{L}}{\partial\theta}` (also
often written :math:`\nabla_{\theta_t}\mathcal{L}`) is the partial derivative of
the loss with respect to the parameters, evaluated at the current value of those
parameters.

The learning rate :math:`\alpha` specifies the scale of these parameter updates
with respect to the magnitude of the gradient. Almost all stochastic optimizers
use a fixed learning rate parameter.

In ``downhill``, the learning rate is passed as a keyword argument to
``minimize()``::

  downhill.minimize(..., learning_rate=0.1)

Often the learning rate is set to a very small value---many approaches seem to
start with values around 1e-4. If the learning rate is too large, the
optimization procedure might "bounce around" in the loss landscape because the
parameter steps are too large. If the learning rate is too small, the
optimization procedure might not make progress quickly enough to make training
practical.

.. _momentum:

Momentum
--------

Momentum is a common technique in stochastic gradient optimization algorithms
that seems to accelerate the optimization process in most cases. Intuitively,
momentum avoids "jitter" in the parameters during optimization by smoothing the
estimates of the local gradient information over time. In practice a momentum
method maintains a "velocity" of the most recent parameter steps and combines
these recent individual steps together when making a parameter update.
Mathematically, this is written:

.. math::

   \begin{eqnarray*}
   \nu_{t+1} &=& \mu \nu_t - \alpha \left. \frac{\partial\mathcal{L}}{\partial\theta} \right|_{\theta_t} \\
   \theta_{t+1} &=& \theta_t + \nu_{t+1}
   \end{eqnarray*}

where the symbols are the same as above, and additionally :math:`\nu` describes
the "velocity" of parameter :math:`\theta`, and :math:`\mu` is the momentum
hyperparameter. The gradient computations using momentum are exactly the same as
when not using momentum; the only difference is the accumulation of recent
updates in the "velocity."

In ``downhill``, the momentum value is passed as a keyword argument to
``minimize()``::

  downhill.minimize(..., momentum=0.9)

Typically momentum is set to a value in :math:`[0, 1)`---when set to 0, momentum
is disabled, and when set to values near 1, the momentum is very high, requiring
several consecutive parameter updates in the same direction to change the
parameter velocity.

In many problems it is useful to set the momentum to a surprisingly large value,
sometimes even to values greater than 0.9. Such values can be especially
effective with a relatively small learning rate.

If the momentum is set too low, then parameter updates will be more noisy and
optimization might take longer to converge, but if the momentum is set too high,
the optimization process might diverge entirely.

Nesterov Momentum
-----------------

More recently, a newer momentum technique has been shown to be even more
performant than "traditional" momentum. This technique was originally proposed
by Y. Nesterov and effectively amounts to computing the momentum value at a
different location in the parameter space, namely the location where the
momentum value would have placed the parameter after the current update:

.. math::
   \begin{eqnarray*}
   \nu_{t+1} &=& \mu \nu_t - \alpha \left.
      \frac{\partial\mathcal{L}}{\partial\theta}\right|_{\theta_t + \mu\nu_t} \\
   \theta_{t+1} &=& \theta_t + \nu_{t+1}
   \end{eqnarray*}

Note that the partial derivative is evaluated at :math:`\theta_t + \mu\nu_t`
instead of at :math:`\theta_t`. The intuitive rationale for this change is that
if the momentum would have produced an "overshoot," then the gradient at this
overshot parameter value would point backwards, toward the previous parameter
value, which would thus help correct oscillations during optimization.

To use Nesterov-style momentum, use either the :class:`NAG
<downhill.first_order.NAG>` optimizer (which uses plain stochastic gradient
descent with Nesterov momentum), or specify ``nesterov=True`` in addition to
providing a nonzero ``momentum`` value when minimizing your loss::

  downhill.minimize(..., momentum=0.9, nesterov=True)

.. _gradient-clipping:

Gradient Clipping
-----------------

Sometimes during the execution of a stochastic optimization routine---and
particularly at the start of optimization, when the problem parameters are far
from their optimal values---the gradient of the loss with respect to the
parameters can be extremely large. In these cases, taking a step that is
proportional to the magnitude of the gradient can actually be harmful, resulting
in an unpredictable parameter change.

To prevent this from happening, but still preserve the iterative loss
improvements when parameters are in a region with "more reasonable" gradient
magnitudes, ``downhill`` implements two forms of "gradient clipping."

The first gradient truncation method rescales the entire gradient vector if its
L2 norm exceeds some threshold. This is accomplished using the
``max_gradient_norm`` hyperparameter::

  downhill.minimize(..., max_gradient_norm=1)

The second gradient truncation method clips each element of the gradient vector
individually. This is accomplished using the ``max_gradient_elem``
hyperparameter::

  downhill.minimize(..., max_gradient_elem=1)

In both cases, gradients that are extremely large will still point in the
correct direction, but their magnitudes will be rescaled to avoid steps that are
too large. Gradients with values smaller than the thresholds (presumably,
gradients near an optimum will be small) will not be affected. In both cases,
the strategy of taking small steps proportional to the gradient seems to work.

.. _optimizing-iteratively:

Optimizing Iteratively
----------------------

The :func:`downhill.minimize` function is actually just a thin wrapper over the
underlying :func:`downhill.Optimizer.iterate` method, which you can use directly
if you want to do something special during training::

  opt = downhill.build('rmsprop', loss=loss, inputs=[x, y])
  for tm, vm in opt.iterate(train=[sizes, prices], momentum=0.9):
      print('training loss:', tm['loss'])
      print('most recent validation loss:', vm['loss'])

Here, we've constructed an :class:`Optimizer <downhill.base.Optimizer>` object,
and we're using it to manually step through the optimization procedure.

Optimizers yield a pair of dictionaries after each optimization epoch; these
dictionaries provide information about the performance of the optimization
procedure. The keys and values in each dictionary give the costs and monitors
that are computed during optimization. There will always be a ``'loss'`` key
that gives the value of the loss function being optimized. In addition, any
monitors that were defined when creating the optimizer will also be provided in
these dictionaries.

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

  downhill.minimize(..., train=batch)

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

  downhill.minimize(..., train=Loader())

There are almost limitless possibilities for using callables to interface with
the optimization process.

.. _training-validation:

Training and Validation
-----------------------

Let's talk for a minute about data. For your typical regression problem, it's
not feasible or even possible to gather *all* of the relevant data---either it's
too expensive to do that, or there might be new data created in the future that
you just don't have any way of predicting.

Given this paucity of data, you're running a risk in using a stochastic
optimizer to solve your problem: the data you have collected might not be
representative of the data that you haven't collected. If this is true, then
your estimates of the loss might be skewed, because these loss estimates are
computed using the "training" data you provide to the optimization algorithm. As
a result, the "optimal" model you find might actually only be optimal with
respect to the data you collected. It might not work well on future data, for
example.

This problem is generally referred to as overfitting_. An optimization algorithm
is designed to minimize the loss on your problem; in some cases you might even
be able to get perfect performance on the data you've collected. But in many
situations, getting perfect performance on your training data is nearly
synonymous with poor performance on future or unseen data!

.. _overfitting: https://en.wikipedia.org/wiki/Overfitting

There are many ways to combat overfitting. One is to tighten your belt and just
gather more data---having more data is a way of ensuring that the data you do
have will be representative of data you will see in the future. Another is to
regularize_ your loss function; this tends to encourage some solutions to your
problem (e.g., solutions with small parameter values) and discourage others
(e.g., solutions that "memorize" outliers). A third way of combatting
overfitting is by gathering a set of "validation" data and stopping the training
process when the performance of your model on the validation set stops improving
(see below for details).

The algorithms in ``downhill`` implement this "early stopping" method; to take
advantage of it, just provide a second set of data when minimizing your loss::

  downhill.minimize(loss,
                    inputs=[x, y],
                    train=[train_sizes, train_prices],
                    valid=[valid_sizes, valid_prices])

.. _regularize: https://en.wikipedia.org/wiki/Regularization_(mathematics)

It's important that the validation dataset not be used during optimization with
early stopping; the idea is that you want to use a small part of the data you've
gathered as a sort of canary_ to guess when the performance of your model will
stop improving when you actually take it out into the world and use it.

.. _canary: https://en.wikipedia.org/wiki/Animal_sentinel#Historical_examples

If you do not specify a validation dataset, the training dataset will also be
used for validation, which effectively disables early stopping.

.. _early-stopping:

Early Stopping
--------------

When you make a call to ``train()`` (or ``itertrain()``), ``theanets`` begins an
optimization procedure.

continue to iterate as long as the training procedure you're using doesn't run
out of patience. So the 50 iterations you're seeing might vary depending on the
model, your dataset, and your training algorithm & parameters. (E.g., the
"sample" trainer only produces one result, because sampling from the training
dataset just happens once, but the SGD-based trainers will run for multiple
iterations.)

For each iteration produced by itertrain using a SGD-based algorithm, the
trainer applies ``train_batches`` gradient updates to the model. Each of these
batches contains ``batch_size`` training examples and computes a single gradient
update. After ``train_batches`` have been processed, the training dataset is
shuffled, so that subsequent iterations might see the same set of batches, but
not in the same order.

The validation dataset is run through the model to test convergence every
``validate_every`` iterations. If there is no progress for ``patience`` of these
validations, then the training algorithm halts and returns.

In theanets, the patience is the number of failed validation attempts
that we're willing to tolerate before seeing any progress. So theanets
will make (``patience`` * ``validate_every``) training updates, checking
(patience) times for improvement before deciding that training should
halt.

In some other tools, the patience is the number of training updates
that we're willing to wait before seeing any progress; these tools
will make (``patience``) training updates, checking (``patience`` /
``validate_every``) times for improvement before deciding that training
should halt. With this definition, you do want to make sure the
validation frequency is smaller than half the patience, to have a good
chance of seeing progress before halting.
