# -*- coding: utf-8 -*-

'''A module of utility functions and other goodies.'''

import click
import datetime
import inspect
import numpy as np
import theano
import theano.tensor as TT


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

    def is_registered(cls, key):
        return key.lower() in cls._registry


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
                         name='{}_{}'.format(param.name, suffix),
                         broadcastable=param.broadcastable)


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


def find_inputs_and_params(node):
    '''Walk a computation graph and extract root variables.

    Parameters
    ----------
    node : Theano expression
        A symbolic Theano expression to walk.

    Returns
    -------
    inputs : list Theano variables
        A list of candidate inputs for this graph. Inputs are nodes in the graph
        with no parents that are not shared and are not constants.
    params : list of Theano shared variables
        A list of candidate parameters for this graph. Parameters are nodes in
        the graph that are shared variables.
    '''
    queue, seen, inputs, params = [node], set(), set(), set()
    while queue:
        node = queue.pop()
        seen.add(node)
        queue.extend(p for p in node.get_parents() if p not in seen)
        if not node.get_parents():
            if isinstance(node, theano.compile.SharedVariable):
                params.add(node)
            elif not isinstance(node, TT.Constant):
                inputs.add(node)
    return list(inputs), list(params)


_detailed_callsite = False


def enable_detailed_callsite_logging():
    '''Enable detailed callsite logging.'''
    global _detailed_callsite
    _detailed_callsite = True


def log(msg, *args, **kwargs):
    '''Log a message to the console.

    Parameters
    ----------
    msg : str
        A string to display on the console. This can contain {}-style
        formatting commands; the remaining positional and keyword arguments
        will be used to fill them in.
    '''
    now = datetime.datetime.now()
    module = 'downhill'
    if _detailed_callsite:
        caller = inspect.stack()[1]
        parts = caller.filename.replace('.py', '').split('/')
        module = '{}:{}'.format(
            '.'.join(parts[parts.index('downhill')+1:]), caller.lineno)
    click.echo(' '.join((
        click.style(now.strftime('%Y%m%d'), fg='blue'),
        click.style(now.strftime('%H%M%S'), fg='cyan'),
        click.style(module, fg='magenta'),
        msg.format(*args, **kwargs),
    )))


def log_param(name, value):
    '''Log a parameter value to the console.

    Parameters
    ----------
    name : str
        Name of the parameter being logged.
    value : any
        Value of the parameter being logged.
    '''
    log('setting {} = {}', click.style(str(name)),
        click.style(str(value), fg='yellow'))
