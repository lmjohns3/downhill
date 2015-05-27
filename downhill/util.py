# -*- coding: utf-8 -*-

'''A module of utility functions and other goodies.'''

import numpy as np
import theano
import theano.tensor as TT


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
    x : float, ndarray, or Theano expression
        Some quantity to cast to floating point.

    Returns
    -------
    x : Theano expression
        A symbolic variable cast as a ``floatX`` value.
    '''
    return TT.cast(x, theano.config.floatX)
