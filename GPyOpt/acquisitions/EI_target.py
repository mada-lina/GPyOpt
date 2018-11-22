# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import get_quantiles

class AcquisitionEI_target(AcquisitionBase):
    """
    Expected improvement acquisition function adapted to the case we want to 
    minimize the deviation to a arget
    
    x_opt = argmin(E{EI_tgt(x)})
    EI_tgt(x) = max(0, Y(x) - Y*(x)) 
    Y(x) = |f(x) - tgt|
    Y*(x) = min(|mu_f(x) - tgt|)
    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter = 0.01, target = None):
        self.optimizer = optimizer
        super(AcquisitionEI_target, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.target = target
        self.jitter = jitter

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        raise NotImplementedError()

    def _compute_acq(self, x):
        """ Computes the Expected Improvement per unit of cost.
        New formula 
        """
        t = self.target
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin_folded_normal(target = t)
        phi_a, Phi_a, a = get_quantiles(self.jitter, fmin, -(m-t), s)
        phi_b, Phi_b, b = get_quantiles(self.jitter, 0, -(m-t), s)
        phi_c, Phi_c, c = get_quantiles(self.jitter, fmin, (m-t), s)
        
        #f_acqu = s * (a * (Phi_a - Phi_b) + c * (Phi_b + Phi_c - 1) + phi_a + phi_c - 2 * phi_b)
        f_acqu = s * (c * (Phi_b + Phi_c - 1) + phi_c -phi_b)
        f_acqu += s * (a * (Phi_a - Phi_b) + phi_a - phi_b)
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        t = self.target
        fmin = self.model.get_fmin_folded_normal(target = t)
        phi_a, Phi_a, a = get_quantiles(self.jitter, fmin, -(m-t), s)
        phi_b, Phi_b, b = get_quantiles(self.jitter, 0, (m-t), s)
        phi_c, Phi_c, c = get_quantiles(self.jitter, fmin, (m-t), s)

        f_acqu = s * (a * (Phi_a - Phi_b) + c * (Phi_b + Phi_c - 1) + phi_a + phi_c - 2 * phi_b)
        df_acqu = dmdx * (Phi_a - 2 * Phi_b - Phi_c + 1) + dsdx * (phi_a + phi_c - 2 * phi_b)
        return f_acqu, df_acqu
