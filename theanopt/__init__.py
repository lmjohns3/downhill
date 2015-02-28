from .base import build, Optimizer
from .first import SGD, NAG
from .second import HF
from .adaptive import RProp, RMSProp, ADADELTA, ESGD


def minimize(loss, params, inputs, train, valid,
             method='rmsprop', updates=(), monitors=(),
             **kwargs):
    '''Minimize a loss function with respect to some symbolic parameters.

    Parameters
    ----------
    loss : Theano expression
        Loss function to minimize. This must be a scalar-valued expression.
    params : list of Theano variables
        Symbolic variables to adjust to minimize the loss.
    inputs : list of Theano variables
        Symbolic variables required to compute the loss.
    train : :class:`Dataset`, ndarray, or callable
        Dataset to use for computing gradient updates.
    valid : :class:`Dataset`, ndarray, or callable
        Dataset to use for validating the minimization process.
    method : str, optional
        Name of the minimization method to use. Must be one of the strings that
        can be passed to :func:`build`. Defaults to ``'rmsprop'``.
    updates : list of update pairs, optional
        A list of pairs providing updates for the internal of the loss
        computation. Normally this is empty, but it can be provided if the loss,
        for example, requires an update to an internal random number generator.
    monitors : dict or sequence of (str, Theano expression) tuples, optional
        Additional values to monitor during optimization. These must be provided
        as either a sequence of (name, expression) tuples, or as a dictionary
        mapping string names to Theano expressions.

    Additional keyword arguments are passed to the optimizer instance.

    Returns
    -------
    train_monitors : dict
        A dictionary mapping monitor names to monitor values. This dictionary
        will always contain the ``'loss'`` key, giving the value of the loss
        evaluated on the training dataset.
    valid_monitors : dict
        A dictionary mapping monitor names to monitor values, evaluated on the
        validation dataset. This dictionary will always contain the ``'loss'``
        key, giving the value of the loss function. Because validation is not
        always computed after every optimization update, these monitor values
        may be "stale"; however, they will always contain the most recently
        computed values.
    '''
    opt = build(method, loss=loss, params=params, monitors=monitors, **kwargs)
    train_monitors = valid_monitors = None
    for train_monitors, valid_monitors in opt.minimize(train, valid):
        pass
    return train_monitors, valid_monitors
