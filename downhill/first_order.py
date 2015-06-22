# -*- coding: utf-8 -*-

'''This module defines first-order gradient descent optimizers.'''

import climate

from .base import Optimizer

logging = climate.get_logger(__name__)

__all__ = ['SGD', 'NAG']


class SGD(Optimizer):
    r'''Basic optimization using stochastic gradient descent.

    Notes
    -----

    A stochastic gradient trainer with momentum :math:`\mu` and learning rate
    :math:`\alpha` updates parameter :math:`\theta` at step :math:`t` by
    blending the current "velocity" :math:`v` with the current gradient
    :math:`\frac{\partial\mathcal{L}}{\partial\theta}`:

    .. math::
        \begin{eqnarray*}
        v_{t+1} &=& \mu v_t - \alpha \frac{\partial\mathcal{L}}{\partial\theta} \\
        \theta_{t+1} &=& \theta_t + v_{t+1}
        \end{eqnarray*}

    Without momentum (i.e., when :math:`\mu = 0`), these updates reduce to
    :math:`\theta_{t+1} = \theta_t - \alpha \frac{\partial\mathcal{L}}{\partial\theta}`,
    which just takes steps downhill according to the the local gradient.

    Adding the momentum term permits the algorithm to incorporate information
    from previous steps as well, which in practice is thought to have the effect
    of incorporating some information about second-order derivatives of the loss
    surface.

    References
    ----------

    .. [Rume86] D. E. Rumelhart, G. E. Hinton, & R. J. Williams. (1986)
       "Learning representations by back-propagating errors". Nature 323
       (6088):533–536. doi:10.1038/323533a0
       http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html
    '''

    def _get_updates_for(self, param, grad):
        yield param, param - self.learning_rate * grad


class NAG(SGD):
    r'''Stochastic gradient optimization with Nesterov momentum.

    This class name is an abbreviation for "Nesterov's Accelerated Gradient."
    Note that the ``momentum`` parameter must be given during optimization for
    Nesterov momentum to be employed; by default ``momentum`` is 0 and so no
    momentum is used.

    Notes
    -----

    The basic difference between NAG and "classical" momentum in SGD
    optimization approaches is that NAG computes the gradients at the position
    in parameter space where "classical" momentum would put us at the *next*
    step. In classical :class:`SGD` with momentum :math:`\mu` and learning rate
    :math:`\alpha`, updates to parameter :math:`p` at step :math:`t` are
    computed by blending the current "velocity" :math:`v` with the current
    gradient :math:`\frac{\partial\mathcal{L}}{\partial p}`:

    .. math::
        \begin{eqnarray*}
        v_{t+1} &=& \mu v_t - \alpha \frac{\partial\mathcal{L}}{\partial p} \\
        p_{t+1} &=& p_t + v_{t+1}
        \end{eqnarray*}

    In contrast, NAG adjusts the update by blending the current "velocity" with
    the gradient at the next step---that is, the gradient is computed at the
    point where the velocity would have taken us:

    .. math::
        \begin{eqnarray*}
        v_{t+1} &=& \mu v_t - \alpha \left.
           \frac{\partial\mathcal{L}}{\partial p}\right|_{p_t + \mu v_t} \\
        p_{t+1} &=& p_t + v_{t+1}
        \end{eqnarray*}

    Again, the difference here is that the gradient is computed at the place in
    parameter space where we would have stepped using the classical technique,
    in the absence of a new gradient.

    In theory, this helps correct for oversteps during learning: If momentum
    would lead us to overshoot, then the gradient at that overshot place will
    point backwards, toward where we came from. See [Suts13]_ for a particularly
    clear exposition of this idea.

    References
    ----------
    .. [Suts13] I. Sutskever, J. Martens, G. Dahl, & G. Hinton. (ICML 2013) "On
       the importance of initialization and momentum in deep learning."
       http://www.cs.toronto.edu/~fritz/absps/momentum.pdf

    .. [Nest83] Y. Nesterov. (1983) "A method of solving a convex programming
       problem with convergence rate O(1/sqr(k))." Soviet Mathematics Doklady,
       27:372–376.
    '''

    def iterate(self, *args, **kwargs):
        kwargs['nesterov'] = True
        return super(NAG, self).iterate(*args, **kwargs)
