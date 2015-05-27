'''This module defines a base class for optimization techniques.'''

import climate
import collections
import numpy as np
import theano
import theano.tensor as TT

logging = climate.get_logger(__name__)


def shared_like(param, suffix, init=0):
    '''Create a Theano shared variable like an existing parameter.

    Parameters
    ----------
    param : Theano variable
        Theano variable to use for shape information.
    suffix : str
        Suffix to append to the parameter's name for the new variable.
    init : float or ndarray, optional
        Initial value of the shared variable. Defaults to 0.

    Returns
    -------
    shared : Theano shared variable
        A new shared variable with the same shape and data type as ``param``.
    '''
    return theano.shared(np.zeros_like(param.get_value()) + init,
                         name='{}_{}'.format(param.name, suffix))


def as_float(x):
    '''Cast a floating point value to a Theano ``floatX`` symbol.

    Parameters
    ----------
    x : float
        A constant.

    Returns
    -------
    x : Theano variable
        A symbolic variable cast as a ``floatX`` value.
    '''
    return TT.cast(x, theano.config.floatX)


def build(method, *args, **kwargs):
    '''Construct an optimizer by name.

    Parameters
    ----------
    method : str
        The name of the optimizer to build.
    args : tuple
        Positional arguments to pass to the optimizer constructor.
    kwargs : dict
        Named arguments to pass to the optimizer constructor.

    Returns
    -------
    optimizer : :class:`Optimizer`
        An optimizer instance.
    '''
    return Optimizer.build(method, *args, **kwargs)


class Registrar(type):
    '''A metaclass that builds a registry of its subclasses.'''

    def __init__(cls, name, bases, dct):
        if not hasattr(cls, '_registry'):
            cls._registry = {}
        else:
            cls._registry[name.lower()] = cls
        super(Registrar, cls).__init__(name, bases, dct)

    def build(cls, key, *args, **kwargs):
        return cls._registry[key.lower()](*args, **kwargs)

    def get_class(cls, key):
        return cls._registry[key.lower()]

    def is_registered(cls, key):
        return key.lower() in cls._registry


class Optimizer(Registrar(str('Base'), (), {})):
    '''An optimizer computes gradient updates to iteratively optimize a loss.

    Parameters
    ----------
    loss : Theano expression
        Loss function to minimize. This must be a scalar-valued expression.
    params : list of Theano variables
        Symbolic variables to adjust to minimize the loss.
    inputs : list of Theano variables
        Symbolic variables required to compute the loss.
    updates : list of update pairs, optional
        A list of pairs providing updates for the internal of the loss
        computation. Normally this is empty, but it can be provided if the loss,
        for example, requires an update to an internal random number generator.
    monitors : dict or sequence of (str, Theano expression) tuples, optional
        Additional values to monitor during optimization. These must be provided
        as either a sequence of (name, expression) tuples, or as a dictionary
        mapping string names to Theano expressions.
    '''

    VALIDATE_EVERY = 10
    '''Default value for the ``validate_every`` parameter.'''

    MIN_IMPROVEMENT = 0
    '''Default value for the ``min_improvement`` parameter.'''

    PATIENCE = 10
    '''Default value for the ``patience`` parameter.'''

    MAX_GRADIENT_NORM = 1000000
    '''Default value for the ``max_gradient_norm`` parameter.'''

    GRADIENT_CLIP = 1000
    '''Default value for the ``gradient_clip`` parameter.'''

    def __init__(self, loss, params, inputs, updates=(), monitors=()):
        self.loss = loss
        self.params = params
        self.inputs = inputs
        self.updates = updates
        if hasattr(updates, 'items') and callable(updates.items):
            self.updates = updates.items()

        self._shapes = [p.get_value(borrow=True).shape for p in self.params]
        self._counts = [np.prod(s) for s in self._shapes]
        self._starts = np.cumsum([0] + self._counts)[:-1]
        self._dtype = self.params[0].get_value().dtype

        self._curr_iter = 0
        self._best_iter = 0
        self._best_loss = 1e100
        self._best_params = [p.get_value().copy() for p in self.params]

        if hasattr(monitors, 'items') and callable(monitors.items):
            monitors = monitors.items()
        self._monitor_exprs = [self.loss]
        self._monitor_names = ['loss']
        for name, monitor in monitors:
            self._monitor_names.append(name)
            self._monitor_exprs.append(monitor)

    def _compile(self):
        '''Compile the Theano functions for evaluating and updating our model.
        '''
        logging.info('compiling evaluation function')
        self.f_eval = theano.function(
            self.inputs, self._monitor_exprs, updates=self.updates)
        logging.info('compiling %s step function', self.__class__.__name__)
        updates = list(self.updates) + list(self._get_updates())
        self.f_step = theano.function(
            self.inputs, self._monitor_exprs, updates=updates)

    def _get_updates(self):
        '''Get parameter update expressions for performing optimization.

        Returns
        -------
        updates : sequence of (parameter, expression) tuples
            A sequence of parameter updates to be applied during optimization.
        '''
        for param, grad in self._differentiate():
            for update in self._get_updates_for(param, grad):
                yield update

    def _get_updates_for(self, param, grad):
        '''Generate some update pairs for the given model parameter.

        Returns
        -------
        updates : sequence of (parameter, expression) tuples
            A sequence of parameter updates to be applied during optimization.
        '''
        raise NotImplementedError

    def _differentiate(self, params=None):
        '''Return a sequence of gradients for our parameters.

        This method applies gradient norm clipping, so if a gradient has a norm
        that exceeds the threshold, it will be rescaled to fit within the norm
        threshold.

        Parameters
        ----------
        params : list of Theano variables, optional
            Return the gradient with respect to these parameters. Defaults to
            all parameters that the optimizer knows about.

        Returns
        -------
        pairs : sequence of (param, grad) tuples
            Generates a sequence of tuples representing each of the parameters
            requested and the corresponding Theano gradient expressions.
        '''
        if params is None:
            params = self.params
        for param, grad in zip(params, TT.grad(self.loss, params)):
            norm = TT.sqrt((grad * grad).sum())
            yield param, TT.clip(
                grad * TT.minimum(1, self.max_gradient_norm / norm),
                -self.gradient_clip, self.gradient_clip)

    def set_params(self, targets):
        '''Set the values of the parameters to the given target values.

        Parameters
        ----------
        targets : sequence of ndarray
            Arrays for setting the parameters of our model.
        '''
        for param, target in zip(self.params, targets):
            param.set_value(target)

    def _log(self, monitors, iteration, label='', suffix=''):
        '''Log the state of the optimizer through the logging system.

        Parameters
        ----------
        monitors : OrderedDict
            A dictionary of monitor names mapped to values. These names and
            values are what is being logged.
        iteration : int
            Optimization iteration that we are logging.
        label : str, optional
            A label for the name of the optimizer creating the log line.
            Defaults to the name of the current class.
        suffix : str, optional
            A suffix to add to the end of the log line, if any.
        '''
        label = label or self.__class__.__name__
        fields = (('{}={:.6f}').format(k, v) for k, v in monitors.items())
        logging.info('%s %i %s%s', label, iteration, ' '.join(fields), suffix)

    def evaluate(self, dataset):
        '''Evaluate the current model parameters on a dataset.

        Parameters
        ----------
        dataset : :class:`Dataset <downhill.dataset.Dataset>`
            A set of data to use for evaluating the model.

        Returns
        -------
        monitors : OrderedDict
            A dictionary mapping monitor names to values. Monitors are
            quantities of interest during optimization---for example, loss
            function, accuracy, or whatever the optimization task requires.
        '''
        values = [self.f_eval(*x) for x in dataset]
        monitors = zip(self._monitor_names, np.mean(values, axis=0))
        return collections.OrderedDict(monitors)

    def _test_patience(self, monitors):
        '''Test whether our patience with optimization has elapsed.

        Parameters
        ----------
        monitors : dict
            A dictionary mapping monitor names to values. The 'loss' key from
            this dictionary will be used to evaluate optimization progress.

        Returns
        -------
        elapsed : bool
            True iff our patience has elapsed and the model is no longer
            improving.
        '''
        self._curr_iter += 1
        marker = ''
        loss = monitors['loss']
        if self._best_loss - loss > self._best_loss * self.min_improvement:
            self._best_loss = loss
            self._best_iter = self._curr_iter
            self._best_params = [p.get_value().copy() for p in self.params]
            marker = ' *'
        self._log(monitors, self._curr_iter - 1, 'validation', marker)
        return self._curr_iter - self._best_iter > self.patience

    def _prepare(self, **kwargs):
        '''Set up properties for optimization.

        This method can be overridden by base classes to provide parameters that
        are specific to a particular optimization technique (e.g., setting up a
        learning rate value).
        '''
        pass

    def iteropt(self, train, valid, **kwargs):
        '''Optimize our loss iteratively using a training and validation dataset.

        This method yields a series of monitor values to the caller. After every
        optimization epoch, a pair of monitor dictionaries is generated: one
        evaluated on the training dataset during the epoch, and another
        evaluated on the validation dataset at the most recent validation epoch.

        The validation monitors might not be updated during every optimization
        iteration; in this case, the most recent validation monitors will be
        yielded along with the training monitors.

        Parameters
        ----------
        train : :class:`Dataset <downhill.dataset.Dataset>`
            A set of training data for computing updates to model parameters.
        valid : :class:`Dataset <downhill.dataset.Dataset>`
            A set of validation data for computing monitor values and
            determining when the loss has stopped improving.

        Returns
        -------
        train_monitors : dict
            A dictionary mapping monitor names to values, evaluated on the
            training dataset.
        valid_monitors : dict
            A dictionary containing monitor values evaluated on the validation
            dataset.
        '''
        self.patience = kwargs.get('patience', self.PATIENCE)
        logging.info('-- patience = %s', self.patience)
        self.validate_every = kwargs.get('validate_every', self.VALIDATE_EVERY)
        logging.info('-- validate_every = %s', self.validate_every)
        self.min_improvement = kwargs.get('min_improvement', self.MIN_IMPROVEMENT)
        logging.info('-- min_improvement = %s', self.min_improvement)
        self.max_gradient_norm = as_float(
            kwargs.get('max_gradient_norm', self.MAX_GRADIENT_NORM))
        logging.info('-- max_gradient_norm = %s', self.max_gradient_norm)
        self.gradient_clip = as_float(
            kwargs.get('gradient_clip', self.GRADIENT_CLIP))
        logging.info('-- gradient_clip = %s', self.gradient_clip)

        self._prepare(**kwargs)
        self._compile()

        iteration = 0
        training = validation = None
        while True:
            if not iteration % self.validate_every:
                try:
                    validation = self.evaluate(valid)
                except KeyboardInterrupt:
                    logging.info('interrupted!')
                    break
                if self._test_patience(validation):
                    logging.info('patience elapsed!')
                    break
            try:
                training = self._step(train)
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break
            iteration += 1
            self._log(training, iteration)
            yield training, validation
        self.set_params(self._best_params)

    def minimize(self, train, valid, **kwargs):
        '''Optimize our loss using a training and validation dataset.

        This method is a thin wrapper over the :func:`iteropt` method. It simply
        exhausts the iterative optimization process and returns the final
        monitor values.

        Parameters
        ----------
        train : :class:`Dataset <downhill.dataset.Dataset>`
            A set of training data for computing updates to model parameters.
        valid : :class:`Dataset <downhill.dataset.Dataset>`
            A set of validation data for computing monitor values and
            determining when the loss has stopped improving.

        Returns
        -------
        train_monitors : dict
            A dictionary mapping monitor names to values, evaluated on the
            training dataset.
        valid_monitors : dict
            A dictionary containing monitor values evaluated on the validation
            dataset.
        '''
        monitors = None
        for monitors in self.iteropt(train, valid, **kwargs):
            pass
        return monitors

    def _step(self, dataset):
        '''Advance the state of the optimizer by one step.

        Parameters
        ----------
        dataset : :class:`Dataset <downhill.dataset.Dataset>`
            A dataset for optimizing the model.

        Returns
        -------
        train_monitors : dict
            A dictionary mapping monitor names to values.
        '''
        values = [self.f_step(*x) for x in dataset]
        return collections.OrderedDict(
            zip(self._monitor_names, np.mean(values, axis=0)))
