# -*- coding: utf-8 -*-

'''This module defines gradient descent optimizers with adaptive learning rates.
'''

import climate
import numpy as np
import theano
import theano.tensor as TT

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .base import Optimizer
from .util import as_float, shared_like

logging = climate.get_logger(__name__)

__all__ = ['RProp', 'RMSProp', 'ADAGRAD', 'ADADELTA', 'ESGD', 'Adam']


class RProp(Optimizer):
    r'''Resilient backpropagation optimizer.

    Notes
    -----

    The RProp method takes small steps in parameter space using local gradient
    information. RProp is unlike "vanilla" first-order techniques like
    :class:`SGD <downhill.first_order.SGD>`, however, because only the signs of
    the gradients are taken into account when making parameter updates. That is,
    the step size for each parameter is independent of the magnitude of the
    gradient for that parameter.

    To accomplish this, RProp maintains a separate learning rate for every
    parameter in the model, and adjusts this learning rate based on the
    consistency of the sign of the gradient over time. Whenever two consecutive
    gradients for a parameter have the same sign, the learning rate for that
    parameter increases, and whenever the signs disagree, the learning rate
    decreases. This has a similar effect to momentum-based stochastic gradient
    methods but effectively maintains parameter-specific learning rates.

    .. math::
        \begin{eqnarray*}
        && \mbox{if } \frac{\partial\mathcal{L}}{\partial p}_{t-1}
           \frac{\partial\mathcal{L}}{\partial p} > 0 \\
        && \qquad \Delta_t = \min (\eta_+\Delta_{t−1}, \Delta_+) \\
        && \mbox{if } \frac{\partial\mathcal{L}}{\partial p}_{t-1}
           \frac{\partial\mathcal{L}}{\partial p} < 0 \\
        && \qquad \Delta_t = \max (\eta_-\Delta_{t−1}, \Delta_-) \\
        && \qquad \frac{\partial\mathcal{L}}{\partial p} = 0 \\
        && p_{t+1} = p_t − \mbox{sgn}\left(
           \frac{\partial\mathcal{L}}{\partial p}\right) \Delta_t
        \end{eqnarray*}

    Here, :math:`s(\cdot)` is the sign function (i.e., returns -1 if its
    argument is negative and 1 otherwise), :math:`\eta_-` and :math:`\eta_+` are
    the amount to decrease (increase) the step size if the gradients disagree
    (agree) in sign, and :math:`\Delta_+` and :math:`\Delta_-` are the maximum
    and minimum step size.

    The implementation here is actually the "iRprop-" variant of RProp described
    in Algorithm 4 from [Igel00]_. This variant resets the running gradient
    estimates to zero in cases where the previous and current gradients have
    switched signs.

    References
    ----------

    .. [Ried92] M. Riedmiller & H. Braun. (1992) "Rprop - A Fast Adaptive
       Learning Algorithm." In Proceedings of the International Symposium on
       Computer and Information Science VII.

    .. [Igel00] C. Igel & M. Hüsken. (2000) "Improving the Rprop Learning
       Algorithm." In Proceedings of the Second International Symposium on
       Neural Computation.
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.1332
    '''

    def _prepare(self,
                 rprop_increase=1.01,
                 rprop_decrease=0.99,
                 rprop_min_step=0,
                 rprop_max_step=100,
                 **kwargs):
        self.step_increase = as_float(rprop_increase)
        self.step_decrease = as_float(rprop_decrease)
        self.min_step = as_float(rprop_min_step)
        self.max_step = as_float(rprop_max_step)
        logging.info('-- rprop_increase = %s', rprop_increase)
        logging.info('-- rprop_decrease = %s', rprop_decrease)
        logging.info('-- rprop_min_step = %s', rprop_min_step)
        logging.info('-- rprop_max_step = %s', rprop_max_step)
        super(RProp, self)._prepare(**kwargs)

    def _get_updates_for(self, param, grad):
        grad_tm1 = shared_like(param, 'grad')
        step_tm1 = shared_like(param, 'step', self.learning_rate.eval())
        test = grad * grad_tm1
        diff = TT.lt(test, 0)
        steps = step_tm1 * (TT.eq(test, 0) +
                            TT.gt(test, 0) * self.step_increase +
                            diff * self.step_decrease)
        step = TT.minimum(self.max_step, TT.maximum(self.min_step, steps))
        grad = grad - diff * grad
        yield param, param - TT.sgn(grad) * step
        yield grad_tm1, grad
        yield step_tm1, step


class ADAGRAD(Optimizer):
    r'''ADAGRAD optimizer.

    Notes
    -----

    The ADAGRAD method uses the same general strategy as all first-order
    stochastic gradient methods, in the sense that these methods make small
    parameter adjustments iteratively using local derivative information.

    The difference with ADAGRAD is that as gradients are computed during each
    parameter update, their squares are accumulated, and this accumulated value
    is used to rescale the global learning rate :math:`\alpha` separately for
    each parameter.

    .. math::
        \begin{eqnarray*}
        g_{t+1} &=& g_t + \left(\frac{\partial\mathcal{L}}{\partial p}\right)^2 \\
        p_{t+1} &=& p_t - \frac{\alpha}{\sqrt{g_{t+1}} + \epsilon}
           \frac{\partial\mathcal{L}}{\partial p}
        \end{eqnarray*}

    Like the other adaptive learning methods, learning method effectively
    maintains a sort of parameter-specific learning rate. Unlike
    :class:`RMSProp` and :class:`ADADELTA`, however, in ADAGRAD, the gradient
    magnitudes accumulate throughout training, which has the effect of scaling
    the learning rate for each parameter, but also effectively anneals the
    learning rate overall as training progresses.

    In this implementation, the scale values are regularized (made less extreme)
    by :math:`\epsilon`, which is specified using the ``regularizer`` parameter.

    References
    ----------

    .. [Duch10] J. Duchi, E. Hazan, & Y. Singer (2010) “Adaptive subgradient
       methods for online leaning and stochastic optimization.” Proc. Conference
       on Learning Theory (COLT).
    '''

    def _prepare(self, regularizer=1e-8, **kwargs):
        self.epsilon = as_float(regularizer)
        logging.info('-- regularizer = %s', regularizer)
        super(ADAGRAD, self)._prepare(**kwargs)

    def _get_updates_for(self, param, grad):
        g2_tm1 = shared_like(param, 'g2_ewma')
        g2_t = g2_tm1 + grad * grad
        delta = grad * self.learning_rate / TT.sqrt(g2_t + self.epsilon)
        yield g2_tm1, g2_t
        yield param, param - delta


class RMSProp(Optimizer):
    r'''RMSProp optimizer.

    Notes
    -----

    The RMSProp method uses the same general strategy as all first-order
    stochastic gradient methods, in the sense that these methods make small
    parameter adjustments iteratively using local derivative information.

    The difference here is that as gradients are computed during each parameter
    update, an exponentially-weighted moving average (EWMA) of gradient
    magnitudes is maintained as well. At each update, the EWMA is used to
    compute the root-mean-square (RMS) gradient value that's been seen in the
    recent past. The actual gradient is normalized by this RMS scaling factor
    before being applied to update the parameters. Intuitively, this makes
    RMSProp take steps near 1 whenever the gradient is of constant magnitude,
    and larger steps whenever the local scale of the gradient starts to
    increase.

    .. math::
        \begin{eqnarray*}
        f_{t+1} &=& \gamma a_t + (1 - \gamma) \frac{\partial\mathcal{L}}{\partial p} \\
        g_{t+1} &=& \gamma g_t + (1 - \gamma) \left(
           \frac{\partial\mathcal{L}}{\partial p}\right)^2 \\
        p_{t+1} &=& p_t - \frac{\alpha}{\sqrt{g_{t+1} - f_{t+1}^2 + \epsilon}}
           \frac{\partial\mathcal{L}}{\partial p}
        \end{eqnarray*}

    Like :class:`RProp`, this learning method effectively maintains a sort of
    parameter-specific momentum value, but this method takes into account both
    the sign and the magnitude of the gradient for each parameter.

    In this algorithm, RMS values are regularized (made less extreme) by
    :math:`\epsilon`, which is specified using the ``rms_regularizer`` keyword
    argument.

    The weight parameter :math:`\gamma` for the EWMA window is computed from the
    ``rms_halflife`` keyword argument, such that the actual EWMA weight varies
    inversely with the halflife :math:`h`: :math:`\gamma = e^{\frac{-\ln
    2}{h}}`.

    The implementation here is taken from [Grav13]_, equations (38)--(45).
    Graves' implementation in particular seems to have introduced the
    :math:`f_t` terms into the RMS computation; these terms appear to act as a
    sort of momentum for the RMS values.

    References
    ----------

    .. [Grav13] A. Graves. (2013) "Generating Sequences With Recurrent Neural
       Networks." http://arxiv.org/abs/1308.0850
    '''

    def _prepare(self, rms_halflife=14, rms_regularizer=1e-8, **kwargs):
        self.ewma = as_float(np.exp(-np.log(2) / rms_halflife))
        self.epsilon = as_float(rms_regularizer)
        logging.info('-- rms_halflife = %s', rms_halflife)
        logging.info('-- rms_regularizer = %s', rms_regularizer)
        super(RMSProp, self)._prepare(**kwargs)

    def _get_updates_for(self, param, grad):
        g1_tm1 = shared_like(param, 'g1_ewma')
        g2_tm1 = shared_like(param, 'g2_ewma')
        g1_t = self.ewma * g1_tm1 + (1 - self.ewma) * grad
        g2_t = self.ewma * g2_tm1 + (1 - self.ewma) * grad * grad
        rms = TT.sqrt(g2_t - g1_t * g1_t + self.epsilon)
        yield g1_tm1, g1_t
        yield g2_tm1, g2_t
        yield param, param - self.learning_rate * grad / rms


class ADADELTA(RMSProp):
    r'''ADADELTA optimizer.

    Notes
    -----

    The ADADELTA method uses the same general strategy as all first-order
    stochastic gradient methods, in the sense that these methods make small
    parameter adjustments iteratively using local derivative information.

    The difference with ADADELTA is that as gradients are computed during each
    parameter update, an exponentially-weighted weighted moving average (EWMA)
    gradient value, as well as an EWMA of recent parameter steps, are maintained
    as well. The actual gradient is normalized by the ratio of the
    root-mean-square (RMS) parameter step size to the RMS gradient magnitude.

    .. math::
        \begin{eqnarray*}
        g_{t+1} &=& \gamma g_t + (1 - \gamma) \left(
           \frac{\partial\mathcal{L}}{\partial p}\right)^2 \\
        v_{t+1} &=& \frac{\sqrt{x_t + \epsilon}}{\sqrt{g_{t+1} + \epsilon}}
           \frac{\partial\mathcal{L}}{\partial p} \\
        x_{t+1} &=& \gamma x_t + (1 - \gamma) v_{t+1}^2 \\
        p_{t+1} &=& p_t - v_{t+1}
        \end{eqnarray*}

    Like :class:`RProp` and the :class:`RMSProp`--:class:`ESGD` family, this
    learning method effectively maintains a sort of parameter-specific momentum
    value. The primary difference between this method and :class:`RMSProp` is
    that ADADELTA additionally incorporates a sliding window of RMS parameter
    step sizes, (somewhat) obviating the need for a learning rate parameter.

    In this implementation, the RMS values are regularized (made less extreme)
    by :math:`\epsilon`, which is specified using the ``rms_regularizer``
    parameter.

    The weight parameter :math:`\gamma` for the EWMA window is computed from the
    ``rms_halflife`` keyword argument, such that the actual EWMA weight varies
    inversely with the halflife :math:`h`: :math:`\gamma = e^{\frac{-\ln
    2}{h}}`.

    References
    ----------

    .. [Zeil12] M. Zeiler. (2012) "ADADELTA: An adaptive learning rate method."
       http://arxiv.org/abs/1212.5701
    '''

    def _get_updates_for(self, param, grad):
        x2_tm1 = shared_like(param, 'x2_ewma')
        g2_tm1 = shared_like(param, 'g2_ewma')
        g2_t = self.ewma * g2_tm1 + (1 - self.ewma) * grad * grad
        delta = grad * TT.sqrt(x2_tm1 + self.epsilon) / TT.sqrt(g2_t + self.epsilon)
        x2_t = self.ewma * x2_tm1 + (1 - self.ewma) * delta * delta
        yield g2_tm1, g2_t
        yield x2_tm1, x2_t
        yield param, param - delta


class ESGD(RMSProp):
    r'''Equilibrated SGD computes a diagonal Hessian preconditioner.

    Notes
    -----

    The ESGD method uses the same general strategy as all first-order
    stochastic gradient methods, in the sense that these methods make small
    parameter adjustments iteratively using local derivative information.

    The difference here is that as gradients are computed during each parameter
    update, an exponentially-weighted moving average (EWMA) of estimates of the
    diagonal of the Hessian (the matrix of second derivatives) is maintained as
    well. At each update, the EWMA is used to compute the root-mean-square (RMS)
    diagonal value that's been seen in the recent past. The actual gradient is
    scaled by the inverse of this diagonal preconditioner before being applied
    to update the parameters. Intuitively, this causes the algorithm to
    "reshape" the loss function in parameter space, such that directions of
    steep gradient (i.e., large diagonal values) and directions of shallow
    gradient (i.e., small diagonal values) are scaled to be approximately the
    same slope.

    The diagonal estimates are computed using a nice trick: A vector :math:`r
    \sim \mathcal{N}(0, 1)` consisting of standard normal values is sampled
    randomly at each update step, and the value of :math:`Hr` is computed
    symbolically. These vector values tend to approximate the diagonal of the
    Hessian. Because :math:`Hr` is itself a vector, the full Hessian :math:`H`
    does not need to be computed or stored.

    .. math::
        \begin{eqnarray*}
        r &\sim& \mathcal{N}(0, 1) \\
        Hr &=& \frac{\partial^2 \mathcal{L}}{\partial^2 p}r \\
        D_{t+1} &=& \gamma D_t + (1 - \gamma) (Hr)^2 \\
        p_{t+1} &=& p_t + - \frac{\alpha}{\sqrt{D_{t+1} + \epsilon}}
           \frac{\partial\mathcal{L}}{\partial p}
        \end{eqnarray*}

    Like :class:`Rprop` and the :class:`ADADELTA`--:class:`RMSProp` family, this
    learning method effectively maintains a sort of parameter-specific learning
    rate for each parameter in the loss.

    In this implementation, :math:`\epsilon` regularizes the RMS values; it is
    is specified using the ``rms_regularizer`` parameter.

    The weight parameter :math:`\gamma` for the EWMA is computed from the
    ``rms_halflife`` keyword argument, such that the actual EWMA weight varies
    inversely with the halflife :math:`h`: :math:`\gamma = e^{\frac{-\ln
    2}{h}}`.

    The primary difference between this implementation and the algorithm
    described in the paper (see below) is the use of an EWMA to decay the
    diagonal values over time, while in the paper the diagonal is divided by the
    training iteration. The EWMA halflife should be set to something reasonably
    large to ensure that this method emulates the method described in the
    original paper.

    References
    ----------

    .. [Daup14] Y. Dauphin, H. de Vries, J. Chung & Y. Bengio. (2014) "RMSProp
       and equilibrated adaptive learning rates for non-convex optimization."
       http://arxiv.org/abs/1502.04390
    '''

    def __init__(self, *args, **kwargs):
        self.rng = RandomStreams()
        super(ESGD, self).__init__(*args, **kwargs)

    def _get_updates_for(self, param, grad):
        D_tm1 = shared_like(param, 'D_ewma')
        Hv = TT.Rop(grad, param, self.rng.normal(param.shape))
        D_t = self.ewma * D_tm1 + (1 - self.ewma) * Hv * Hv
        den = TT.sqrt(D_t) + self.epsilon
        yield D_tm1, D_t
        yield param, param - grad * self.learning_rate / den


class Adam(RMSProp):
    r'''Adam optimizer using unbiased gradient moment estimates.

    Notes
    -----

    The Adam method uses the same general strategy as all first-order
    stochastic gradient methods, in the sense that these methods make small
    parameter adjustments iteratively using local derivative information.

    The difference here is that as gradients are computed during each parameter
    update, exponentially-weighted moving averages (EWMAs) of (1) the first
    moment of the recent gradient values and (2) the second moment of recent
    gradient values are maintained as well. At each update, the step taken is
    proportional to the ratio of the first moment to the second moment.

    .. math::
        \begin{eqnarray*}
        \beta_1^t &=& \beta_1 \lambda^{t}
        f_{t+1} &=& \beta_1^t f_t + (1 - \beta_1^t)
           \frac{\partial\mathcal{L}}{\partial\theta} \\
        g_{t+1} &=& \beta_2 g_t + (1 - \beta_2)
           \left(\frac{\partial\mathcal{L}}{\partial\theta}\right)^2 \\
        \theta_{t+1} &=& \theta_t -
           \frac{f_{t+1} / (1 - \beta_1^t)}{\sqrt{g_{t+1} / (1 - \beta_2)} + \epsilon}
        \end{eqnarray*}

    Like all adaptive optimization algorithms, this optimizer effectively
    maintains a sort of parameter-specific momentum value. It shares with
    :class:`RMSProp` and :class:`ADADELTA` the idea of using an EWMA to track
    recent quantities related to the stochastic gradient during optimization.
    But the Adam method is unique in that it incorporates an explicit
    computation to remove the bias from these estimates.

    In this implementation, :math:`\epsilon` regularizes the RMS values and is
    given using the ``rms_regularizer`` keyword argument. The weight parameters
    :math:`\beta_1` and :math:`\beta_2` for the first and second EWMA windows
    are computed from the ``beta1_halflife`` and ``beta2_halflife`` keyword
    arguments, respectively, such that the actual EWMA weight varies inversely
    with the halflife :math:`h`: :math:`\gamma = e^{\frac{-\ln 2}{h}}`. The
    decay :math:`\lambda` for the :math:`\beta_1` EWMA is provided by the
    ``beta1_decay`` keyword argument.

    The implementation here is taken from Algorithm 1 of [King15]_.

    References
    ----------

    .. [King15] D. Kingma & J. Ba. (ICLR 2015) "Adam: A Method for
       Stochastic Optimization." http://arxiv.org/abs/1412.6980
    '''

    def _prepare(self,
                 beta1_decay=1 - 1e-6,
                 beta1_halflife=7,
                 beta2_halflife=69,
                 **kwargs):
        self.beta1_decay = as_float(beta1_decay)
        self.beta1 = as_float(np.exp(-np.log(2) / beta1_halflife))
        self.beta2 = as_float(np.exp(-np.log(2) / beta2_halflife))
        super(Adam, self)._prepare(**kwargs)

    def _get_updates_for(self, param, grad):
        t_tm1 = theano.shared(np.cast['float32'](0), 't')
        g1_tm1 = shared_like(param, 'g1_ewma')
        g2_tm1 = shared_like(param, 'g2_ewma')
        beta1 = self.beta1 * self.beta1_decay ** t_tm1
        g1_t = beta1 * g1_tm1 + (1 - beta1) * grad
        g2_t = self.beta2 * g2_tm1 + (1 - self.beta2) * grad * grad
        num = g1_t / (1 - beta1)
        den = TT.sqrt(g2_t / (1 - self.beta2))
        yield t_tm1, t_tm1 + 1
        yield g1_tm1, g1_t
        yield g2_tm1, g2_t
        yield param, param - self.learning_rate * num / (den + self.epsilon)
