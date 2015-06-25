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
what ``downhill`` does: given a computational formulation of a loss, the
optimization routines in ``downhill`` can compute a series of ever-better
solutions to your problem.

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
Theano_, a Python framework for describing computation graphs. Theano takes a
bit of getting used to, but we'll walk through the linear regression example
here; if you're curious, there are also lots of good tutorials_ on the Theano
site.

To use Theano with ``downhill``, you need to define `shared variables`_ for each
of the parameters in your model, and `symbolic inputs`_ for the data that you'll
use to evaluate your loss. We'll start with the shared variables::

  import downhill
  import numpy as np
  import theano
  import theano.tensor as TT

  m = theano.shared(np.ones((1, ), 'f'), name='m')
  b = theano.shared(np.zeros((1, ), 'f'), name='b')

This sets up a vector with one 1 for :math:`m`, and a vector with one 0 for
:math:`b`. The values contained inside these shared variables will be adjusted
automatically by the optimization algorithms in ``downhill``.

Next, you need to define symbols that represent the data needed to compute
the loss::

  x = TT.vector('x')
  y = TT.vector('y')

These symbolic vectors represent the inputs---the house sizes :math:`[x_1 \dots
x_N]` and prices :math:`[y_1 \dots y_N]`---needed to compute the loss. Finally,
having created all of these symbolic variables, you can define the loss itself::

  loss = TT.sqr(m * x + b - y).sum()

This tells Theano to multiply the data vector ``x`` by the value stored in the
shared ``m`` variable, add the value stored in the shared ``b`` variable, and
then subtract the data vector ``y``. Then that vector gets squared elementwise,
and all of the components of the result get summed up to produce the loss.

Note that none of these operations have actually been computed; instead, you've
instructed Theano *how* to compute the loss, if you were to give it some values
for ``x`` and ``y``. This is the bizarre thing about Theano: it looks like
you're computing things, but you're actually just telling the computer how to
compute things in the future.

.. _Theano: http://deeplearning.net/software/theano/
.. _tutorials: http://deeplearning.net/software/theano/tutorial/index.html
.. _shared variables: http://deeplearning.net/software/theano/tutorial/examples.html#using-shared-variables
.. _symbolic inputs: http://deeplearning.net/software/theano/tutorial/adding.html

.. _minimizing-loss:

Minimizing a Loss
=================

The ``downhill`` package provides a single high-level function,
:func:`downhill.minimize`, that can be used as a black-box optimizer for losses.
In addition, there are lower-level calls that provide more control over the
interaction between your code and ``downhill``. First, we'll look at the
high-level minimize function, then we'll talk about what happens under the hood.

Once you've defined your loss using Theano, you can minimize it with a single
function call. Here, we'll minimize the loss defined above::

  downhill.minimize(loss, [sizes, prices], inputs=[x, y])

You just specify the loss to minimize, provide some data to use for computing
the loss, and identify the symbolic inputs that the loss requires. The
``downhill`` code will select an optimization algorithm (the default is
currently :class:`RMSProp <downhill.adaptive.RMSProp>`), identify shared
variables in the loss that need optimization, and run the optimization process
to completion. After the minimization has finished, the shared variables in your
loss will be updated to their optimal values. You can retrieve their values
using any of the methods of `shared variables`_::

  m_value, b_value = m.get_value(), b.get_value()

There is much to say about providing data---see :ref:`providing-data` for more
information---but briefly, the data you will need to provide is typically a list
of ``numpy`` arrays of the measurements you've made for your problem. For the
house price regression example, the arrays for house size and house price might
be set up like this::

  sizes = np.array([1200, 2013, 8129, 2431, 2211])
  prices = np.array([103020, 203310, 3922013, 224321, 449020])

.. _training-validation:

Training and Validation
-----------------------

You might have noticed that the formulation of the loss given at the top of this
guide contains a sum over all of the data points that you've observed
:math:`(x_i, y_i)`. (For the house price example, these data are stored in the
``sizes`` and ``prices`` arrays.) This is a very common state of affairs for
many problems: the loss is computed thanks to observed data.

But for a typical regression problem, it's not feasible or even possible to
gather *all* of the relevant data---either it's too expensive to do that, or
there might be new data created in the future that you don't have any way of
predicting.

Given this paucity of data, you're running a risk in using a stochastic
optimizer to solve your problem: the data that you *have* collected might not be
representative of the data that you *haven't* collected! If the data you
collected are quite different from the "true" data out there in the world, then
when you optimize your loss, the optimal model might be skewed toward your
dataset, and your model might not perform well on new, "unseen" data.

This problem is generally referred to as overfitting_ and is a risk with many
types of models. Generally the risk of overfitting increases with the complexity
of your model, and also increases when you don't have a lot of data.

There are many ways to combat overfitting:

- You can tighten your belt and gather more data, which increases the chance
  that the data you do have will be representative of data you don't yet have.

- You can regularize_ your loss; this tends to encourage some solutions to your
  problem (e.g., solutions with small parameter values) and discourage others
  (e.g., solutions that "memorize" outliers).

- You can also set aside a bit of the data you've collected as a "validation"
  set. You can use this set to stop the optimization process when the
  performance of your model on the validation set stops improving---this is
  known as "early stopping."

Collecting more data is almost always a good idea, as long as you can afford to
do so (whether in terms of time, monetary cost, etc.)---but ``downhill`` can't
help you with that. And while it can often be a good idea to incorporate
regularizers into your loss, doing so is something of an art and remains outside
the scope of ``downhill``.

.. _overfitting: https://en.wikipedia.org/wiki/Overfitting
.. _regularize: https://en.wikipedia.org/wiki/Regularization_(mathematics)

.. _early-stopping:

Early Stopping
--------------

The algorithms in ``downhill`` implement the "early stopping" regularization
method. To take advantage of it, just provide a second set of data when
minimizing your loss::

  downhill.minimize(loss, [sizes, prices], [valid_sizes, valid_prices])

Here we'll assume that you've gathered another few sizes and prices and put them
in a new pair of ``numpy`` arrays. In practice, the validation dataset can also
just be a small bit (10% or so) of the training data you've collected. Either
way, it's important to make sure the validation data is disjoint from the
training data, to ensure the most accurate predictions on unseen data. The idea
is that you want to use a small part of the data you've gathered as a sort of
canary_ to guess when the performance of your model will be good when you
actually take it out into the world and use it.

.. _canary: https://en.wikipedia.org/wiki/Animal_sentinel#Historical_examples

The early stopping method will cause optimization to halt when the loss stops
improving on the validation dataset. If you do not specify a validation dataset,
the training dataset will also be used for validation, which effectively
disables early stopping---that is, optimization will halt whenever the loss
computed on the training dataset stops improving.

To understand this better, we'll take a look at the lower-level API provided by
``downhill``.

.. _iterative-optimization:

Iterative Optimization
----------------------

The :func:`downhill.minimize` function is actually just a wrapper that performs
a few common lower-level tasks to optimize your loss. These tasks include:

- creating :class:`datasets <downhill.dataset.Dataset>` to wrap your data,
- creating an :class:`Optimizer <downhill.base.Optimizer>`, and
- running the optimizer to completion.

You can perform these tasks yourself to retain more control over the
optimization process, but even if you don't, it's useful to follow the process
to know how it works. In practice it can often be useful to call the
:func:`iterate() <downhill.base.Optimizer.iterate>` method yourself, because it
gives you access to the state of the optimizer at each step.

To learn more about this, have a look at the following example::

  opt = downhill.build('rmsprop', loss=loss, inputs=[x, y])
  train = downhill.Dataset([sizes, prices])
  valid = downhill.Dataset([valid_sizes, valid_prices])
  for tm, vm in opt.iterate(train, valid):
      print('training loss:', tm['loss'])
      print('most recent validation loss:', vm['loss'])

This code constructs an :class:`Optimizer <downhill.base.Optimizer>` object
(specifically, an :class:`RMSProp optimizer <downhill.adaptive.RMSProp>`), wraps
the input data with a :class:`Dataset <downhill.dataset.Dataset>`, and then
steps through the optimization process iteratively.

Notice that after each iteration, the optimizer yields a pair of dictionaries to
the caller: the first dictionary contains measured values of the loss on the
training data during that iteration, and the second contains measured values of
the loss on the validation data.

The keys and values in each of these dictionaries give the costs and monitors
that are computed during optimization. There will always be a ``'loss'`` key in
each dictionary that gives the value of the loss function being optimized. In
addition, any :ref:`monitor values <monitoring>` that were defined when creating
the optimizer will also be provided in these dictionaries.

.. _batches-epochs:

Batches and Epochs
------------------

During each iteration, the optimizer instance processes training data in small
pieces called "mini-batches"; each mini-batch is used to compute a gradient
estimate for the loss, and the parameters are updated by a small amount. After a
fixed number of mini-batches have been processed, the ``iterate`` method yields
the loss dictionaries to the caller.

Each group of parameter updates processed during a single iteration is called an
"epoch." After a fixed number of epochs have taken place, the loss is then
evaluated using a fixed number of mini-batches from the validation dataset, and
this result is saved as the validation dictionary after every epoch until the
next validation happens.

Optimization epochs continue to occur, with occasional validations, until the loss
on the validation dataset fails to make sufficient progress for long enough.
Optimization halts at that point.

There are a number of hyperparameters involved in this process, which can be
tuned for the best performance on your problem.

.. _tuning:

Tuning
======

The ``downhill`` package provides several ways of tuning the optimization
process. There are many different settings for mini-batch optimization and
validation, many optimization algorithms are available, and there are also
several common learning hyperparameters that might require tuning.

.. _batch-parameters:

Batch Parameters
----------------

All algorithms in ``downhill`` provide early stopping and use :ref:`epoch-based
optimization <batches-epochs>` as described above. This process is controlled by
a number of parameters that can be tweaked for your optimization problem.

The size of a minibatch is controlled using the ``batch_size`` parameter when
you create a :class:`Dataset <downhill.dataset.Dataset>`. To build mini-batches
containing 3 pieces of data, for example::

    train = downhill.Dataset([sizes, prices], batch_size=3)

If you call the high-level :func:`downhill.minimize` method directly, you can
pass ``batch_size`` to it directly::

    downhill.minimize(loss, [sizes, prices], batch_size=3)

The number of mini-batches that are processed during a single training epoch is
controlled by the ``iteration_size`` parameter when constructing a ``Dataset``::

    train = downhill.Dataset([sizes, prices], iteration_size=10)

This will ensure that one iteration loop over the training dataset will produce
10 mini-batches. If you have fewer than ``batch_size`` times ``iteration_size``
pieces of data, the ``Dataset`` class will loop over your data multiple times to
ensure that the desired number of batches is processed. (The ``Dataset`` class
also handles shuffling your data as needed during iteration, to avoid issues
that can come up when presenting data to the model in a fixed order.)

If you call the high-level :func:`downhill.minimize` method, the numbers of
training and validation mini-batches processed per epoch are set using the
``train_batches`` and ``valid_batches`` parameters, respectively::

  downhill.minimize(..., train_batches=10, valid_batches=8)

Finally, a validation takes place after a fixed number of training epochs have
happened. This number is set using the ``validate_every`` parameter; for
example, to validate the loss every 5 training epochs::

  downhill.minimize(..., validate_every=5)

If you are processing data using the lower-level API, the ``validate_every``
parameter is passed directly to :func:`iterate()
<downhill.base.Optimizer.iterate>`::

  for tm, vm in opt.iterate(..., validate_every=5):
      # ...

.. _patience-improvement:

Patience and Improvement
------------------------

The training process halts if there is "insufficient" progress on the validation
loss for "long enough." The precise meanings of these terms are given by the
``min_improvement`` and ``patience`` parameters, respectively.

The ``min_improvement`` parameter specifies the minimum relative improvement of
the validation loss that counts as progress in the optimization. If
``min_improvement`` is set to 0, for example, then any positive improvement in
the validation loss will count as progress, while if ``min_improvement`` is set
to 0.1, then the validation loss must improve by 10% relative to the current
best validation loss before the validation attempt counts as progress.

The ``patience`` parameter specifies the number of failed validation attempts
that you are willing to tolerate before seeing any progress. If ``patience`` is
set to 0, for instance, then optimization will halt as soon as a validation
attempt fails to make ``min_improvement`` relative loss improvement over the
best validation loss so far. If ``patience`` is set to 3, then optimization will
continue through three failed validation attempts, but if the fourth validation
attempt fails, then optimization will halt.

These parameters can be set either on a call to the high-level
:func:`downhill.minimize` function::

  downhill.minimize(..., patience=3, min_improvement=0.1)

or when calling :func:`iterate() <downhill.base.Optimizer.iterate>`::

  for tm, vm in opt.iterate(..., patience=3, min_improvement=0.1):
      # ...

.. _algorithm:

Optimization Algorithms
-----------------------

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

or pass the algorithm name to build an :class:`Optimizer
<downhill.base.Optimizer>` instance::

  opt = downhill.build('adadelta', ...)

Different algorithms have different performance characteristics, different
numbers of hyperparameters to tune, and different suitability for particular
problems. In general, several of the the adaptive procedures seem to work well
across different problems, particularly :class:`Adam <downhill.adaptive.Adam>`,
:class:`ADADELTA <downhill.adaptive.ADADELTA>`, and :class:`RMSProp
<downhill.adaptive.RMSProp>`. :class:`NAG <downhill.first_order.NAG>` also seems
to work quite well, but can sometimes take longer to converge.

Many of these algorithms, being based on stochastic gradient descent, rely on a
common set of hyperparameters that control the speed of convergence and the
reliability of the optimization process over time; these parameters are
discussed next.

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
:func:`downhill.minimize`::

  downhill.minimize(..., learning_rate=0.1)

Often the learning rate is set to a very small value---many approaches seem to
start with values around 1e-4. If the learning rate is too large, the
optimization procedure might "bounce around" in the loss landscape because the
parameter steps are too large. If the learning rate is too small, the
optimization procedure might not make progress quickly enough to make
optimization practical.

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
:func:`downhill.minimize`::

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

.. _providing-data:

Providing Data
==============

As described above, you'll often need to provide data to ``downhill`` so that
you can compute the loss and optimize the parameters for your problem. There are
two ways of passing data to ``downhill``: using arrays and using callables.

.. _data-using-arrays:

Using Arrays
------------

A fairly typical use case for optimizing a loss for a small-ish problem is to
construct a ``numpy`` array containing the data you have::

  dataset = np.load(filename)
  downhill.minimize(..., train=dataset)

Sometimes the data available for optimizing a loss exceeds the available
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
provide data to the optimization algorithm.

.. _data-using-callables:

Using Callables
---------------

Instead of an array of data, you can provide a callable for a :class:`Dataset
<downhill.dataset.Dataset>`. This callable must take no arguments and must
return a list of ``numpy`` arrays of the proper shape for your loss.

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

.. _monitoring:

Monitoring
==========

Sometimes while optimizing a loss, it can be helpful to "see inside" the model.
In a model with a sparsity regularizer, for example, having some idea of the
current sparsity of the model can help diagnose when the model is "too sparse."

In ``downhill`` you can provide a series of *monitors* during optimization that
satisfy this need. Monitors must be a series of named Theano expressions that
evaluate to scalars; this can be provided as a dictionary that maps names to
expressions, or as a list of (name, expression) ordered pairs.

Suppose you want to monitor the slope and intercept values that your model is
computing as it works its way through the house price modeling task. You can
provide monitors for these quantities as follows::

  downhill.minimize(
      loss,
      [sizes, prices],
      inputs=[x, y],
      monitors=[
          ('m', m.sum()),
          ('b', b.sum()),
      ])

The Theano expressions here are sums because the ``m`` and ``b`` shared
variables are actually arrays of shared variables. (This also helps generalize
the regression loss to situations where you might have multiple independent
variables, like house size and number of bedrooms.) If you preferred to provide
the monitor values as a dictionary, it would look like::

  downhill.minimize(
      loss,
      [sizes, prices],
      inputs=[x, y],
      monitors=dict(m=m.sum(), b=b.sum()))

Note that if you construct an :class:`Optimizer <downhill.base.Optimizer>`
directly, then you need to pass the monitors when you create your optimizer
instance::

  opt = downhill.build(
      'nag', loss=loss, inputs=[sizes, prices],
      monitors=dict(m=m.sum(), b=b.sum()))

Gradients
---------

Sometimes when setting parameters like ``learning_rate`` and
``max_gradient_norm``, it can be quite useful to see how large the gradients of
your model are. These quantities can be included in the monitors easily by
setting the ``monitor_gradients`` flag::

  downhill.minimize(
      loss,
      [sizes, prices],
      inputs=[x, y],
      monitor_gradients=True)

This will include one monitor for each parameter in your model, indicating the
squared L2 norm of the gradient (averaged across mini-batches).

More Information
================

This concludes the ``downhill`` guide! Have a good time harnessing the power of
your GPU to optimize your scalar losses!

If you need more information or just want to discuss things, sign up for the
`mailing list`_, and check out the project page at github_.

.. _mailing list: https://groups.google.com/forum/#!forum/downhill-users
.. _github: https://github.com/lmjohns3/downhill
