# -*- coding: utf-8 -*-

'''This module defines first-order gradient descent optimizers.'''

import climate

from .base import Optimizer
from .util import as_float, shared_like

logging = climate.get_logger(__name__)

__all__ = ['SGD', 'NAG']


class SGD(Optimizer):
    r'''Optimize using stochastic gradient descent with momentum.

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
    '''

    def _get_updates_for(self, param, grad):
        vel_tm1 = shared_like(param, 'vel')
        vel_t = self.momentum * vel_tm1 - self.learning_rate * grad
        yield vel_tm1, vel_t
        yield param, param + vel_t


class NAG(SGD):
    r'''Optimize using Nesterov's Accelerated Gradient (NAG).

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
    point backwards, toward where we came from. See [1]_ for details on this idea.

    References
    ----------
    .. [1] I. Sutskever, J. Martens, G. Dahl, & G. Hinton. (ICML 2013) "On the
       importance of initialization and momentum in deep learning."
       http://jmlr.csail.mit.edu/proceedings/papers/v28/sutskever13.pdf
    '''

    def _prepare(self, **kwargs):
        super(NAG, self)._prepare(**kwargs)
        self.nesterov = True

    def _get_updates_for(self, param, grad):
        # see https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617
        vel_tm1 = shared_like(param, 'vel')
        vel_t = self.momentum * vel_tm1 - self.learning_rate * grad
        yield vel_tm1, vel_t
        yield param, param + self.momentum * vel_t - self.learning_rate * grad
