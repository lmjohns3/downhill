# -*- coding: utf-8 -*-

'''This module defines miscellaneous optimizers.'''

import climate
import numpy as np
import scipy.optimize
import theano.tensor as TT

from .base import Optimizer

logging = climate.get_logger(__name__)


class Scipy(Optimizer):
    '''General trainer for neural nets using ``scipy``.

    This class serves as a wrapper for the optimization algorithms provided in
    `scipy.optimize.minimize`_. The following algorithms are available in this
    trainer:

    - ``bfgs``
    - ``cg``
    - ``dogleg``
    - ``newton-cg``
    - ``trust-ncg``

    In general, these methods require two types of computations in order to
    minimize a cost function: evaluating the cost function for a specific
    setting of model parameters, and computing the gradient of the cost function
    for a specific setting of model parameters. Both of these computations are
    implemented by the ``theanopt`` package and may, if you have a GPU, involve
    computing values on the GPU.

    However, all of the optimization steps that might be performed once these
    two types of values are computed will not be handled on the GPU, since
    ``scipy`` is not capable of using the GPU. This might or might not influence
    the absolute time required to optimize a model, depending on the ratio of
    time spent computing cost and gradient values to the time spent computing
    parameter updates.

    For more information about these optimization methods, please see the `Scipy
    documentation`_.

    .. _scipy.optimize.minimize: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    .. _Scipy documentation: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    '''

    METHODS = ('bfgs', 'cg', 'dogleg', 'newton-cg', 'trust-ncg')

    def __init__(self, method, *args, **kwargs):
        super(Scipy, self).__init__(*args, **kwargs)

        self.method = method

    def compile(self):
        super(Scipy, self).compile()
        logging.info('compiling gradient function')
        self.f_grad = theano.function(self.inputs, TT.grad(self.loss, self.params))

    def flat_to_arrays(self, x):
        '''Convert a parameter vector to a sequence of parameter arrays.

        Parameters
        ----------
        flat : ndarray
            A one-dimensional numpy array containing flattened parameter values
            for all parameters in our model.

        Returns
        -------
        arrays : sequence of ndarray
            Values of the parameters in our model.
        '''
        x = x.astype(self._dtype)
        return [x[o:o+n].reshape(s) for s, o, n in
                zip(self._shapes, self._starts, self._counts)]

    def arrays_to_flat(self, arrays):
        '''Convert a sequence of parameter arrays to a vector.

        Parameters
        ----------
        arrays : sequence of ndarray
            Values of the parameters in our model.

        Returns
        -------
        flat : ndarray
            A one-dimensional numpy array containing flattened parameter values
            for all parameters in our model.
        '''
        x = np.zeros((sum(self._counts), ), self._dtype)
        for arr, o, n in zip(arrays, self._starts, self._counts):
            x[o:o+n] = arr.ravel()
        return x

    def function_at(self, x, dataset):
        '''Compute the value of the loss function at given parameter values.

        Parameters
        ----------
        x : ndarray
            An array of parameter values to set our model at.
        dataset : :class:`Dataset <theanopt.dataset.Dataset>`
            A set of data over which to compute our loss function.

        Returns
        -------
        loss : float
            Scalar value of the loss function, evaluated at the given parameter
            settings, using the given dataset.
        '''
        self.set_params(self.flat_to_arrays(x))
        return self.evaluate(dataset)['loss']

    def gradient_at(self, x, dataset):
        '''Compute the gradients of the loss function at given parameter values.

        Parameters
        ----------
        x : ndarray
            An array of parameter values to set our model at.
        dataset : :class:`Dataset <theanopt.dataset.Dataset>`
            A set of data over which to compute our gradients.

        Returns
        -------
        gradients : ndarray
            A vector of gradient values, of the same dimensions as `x`.
        '''
        self.set_params(self.flat_to_arrays(x))
        grads = [[] for _ in range(len(self.params))]
        for x in dataset:
            for i, g in enumerate(self.f_grad(*x)):
                grads[i].append(np.asarray(g))
        return self.arrays_to_flat([np.mean(g, axis=0) for g in grads])

    def step(self, dataset):
        '''Advance the state of the model by one training step.

        Parameters
        ----------
        dataset : :class:`Dataset <theanopt.dataset.Dataset>`
            A dataset for training the model.

        Returns
        -------
        training : dict
            A dictionary mapping monitor names to values.
        '''
        res = scipy.optimize.minimize(
            fun=self.function_at,
            jac=self.gradient_at,
            x0=self.arrays_to_flat(self._best_params),
            args=(dataset, ),
            method=self.method,
            options=dict(maxiter=self.validate_every),
        )
        self.set_params(self.flat_to_arrays(res.x))
        return self.evaluate(dataset)
