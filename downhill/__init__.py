from .adaptive import *
from .base import build, Optimizer
from .dataset import Dataset
from .first_order import *

__version__ = '0.3.1'


def minimize(loss, train, valid=None, params=None, inputs=None, algo='rmsprop',
             updates=(), monitors=(), monitor_gradients=False, batch_size=32,
             train_batches=None, valid_batches=None, **kwargs):
    '''Minimize a loss function with respect to some symbolic parameters.

    Additional keyword arguments are passed to the underlying :class:`Optimizer
    <downhill.base.Optimizer>` instance.

    Parameters
    ----------
    loss : Theano expression
        Loss function to minimize. This must be a scalar-valued expression.
    train : :class:`Dataset <downhill.dataset.Dataset>`, ndarray, or callable
        Dataset to use for computing gradient updates.
    valid : :class:`Dataset <downhill.dataset.Dataset>`, ndarray, or callable, optional
        Dataset to use for validating the minimization process. The training
        dataset is used if this is not provided.
    params : list of Theano variables, optional
        Symbolic variables to adjust to minimize the loss. If not given, these
        will be computed automatically by walking the computation graph.
    inputs : list of Theano variables, optional
        Symbolic variables required to compute the loss. If not given, these
        will be computed automatically by walking the computation graph.
    algo : str, optional
        Name of the minimization algorithm to use. Must be one of the strings
        that can be passed to :func:`build`. Defaults to ``'rmsprop'``.
    updates : list of update pairs, optional
        A list of pairs providing updates for the internal of the loss
        computation. Normally this is empty, but it can be provided if the loss,
        for example, requires an update to an internal random number generator.
    monitors : dict or sequence of (str, Theano expression) tuples, optional
        Additional values to monitor during optimization. These must be provided
        as either a sequence of (name, expression) tuples, or as a dictionary
        mapping string names to Theano expressions.
    monitor_gradients : bool, optional
        If True, add monitors to log the norms of the parameter gradients during
        optimization. Defaults to False.
    batch_size : int, optional
        Size of batches provided by datasets. Defaults to 32.
    train_batches : int, optional
        Number of batches of training data to iterate over during one pass of
        optimization. Defaults to None, which uses the entire training dataset.
    valid_batches : int, optional
        Number of batches of validation data to iterate over during one pass of
        validation. Defaults to None, which uses the entire validation dataset.

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
    if not isinstance(train, Dataset):
        train = Dataset(
            train,
            name='train',
            batch_size=batch_size,
            iteration_size=train_batches,
        )
    if valid is not None and not isinstance(valid, Dataset):
        valid = Dataset(
            valid,
            name='valid',
            batch_size=batch_size,
            iteration_size=valid_batches,
        )
    return build(
        algo,
        loss=loss,
        params=params,
        inputs=inputs,
        updates=updates,
        monitors=monitors,
        monitor_gradients=monitor_gradients,
    ).minimize(train, valid, **kwargs)
