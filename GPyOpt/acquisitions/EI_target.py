# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import get_quantiles, folded_normal
import numpy as np

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

    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter = 0.01, target = None, nb_output=1):
        self.optimizer = optimizer
        super(AcquisitionEI_target, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients, nb_output=nb_output)
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
        fmin = self.model.get_fmin(target = t, fold = True)
        fmin = np.repeat(np.atleast_1d(fmin)[np.newaxis, :], len(m), 0)
        t = np.repeat(np.atleast_1d(t)[np.newaxis, :], len(m), 0)
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

    def _compute_acq_novar(self, x):
        """
        Computes the acquisition function without the uncertainty part i.e. the expected value of the fom
        """
        m, s = self.model.predict(x)
        m_folded, _ = folded_normal(m - np.repeat(np.atleast_1d(self.target)[np.newaxis, :], len(m), 0), s)
        return np.average(np.atleast_1d(m_folded), 1)[:, np.newaxis]


    def _compute_acq_splitted(self, x):
        """
        Computes the two parts (expected value, std) used in the acqu function 
        """
        m, s = self.model.predict(x)
        m_folded, s_folded = folded_normal(m - np.repeat(np.atleast_1d(self.target)[np.newaxis, :], len(m), 0), s)
        m_exp = np.average(np.atleast_1d(m_folded), 1)[:, np.newaxis]
        n_var = np.shape(m_exp)[1] if np.ndim(m_exp)> 1 else 1
        s_exp = np.sqrt(1/n_var * np.average(np.atleast_1d(np.square(s_folded)), 1))[:, np.newaxis]
        return m_exp, s_exp 