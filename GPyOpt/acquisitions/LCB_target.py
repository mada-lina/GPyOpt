# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import folded_normal
import numpy as np


class AcquisitionLCB_target(AcquisitionBase):
    """
    GP-Lower Confidence Bound acquisition function
    U

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: does not allow to be used with cost
    #TODO: Implement gradient
    """

    analytical_gradient_prediction = False 

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2, target = None, nb_output=1):
        self.optimizer = optimizer
        super(AcquisitionLCB_target, self).__init__(model, space, optimizer, nb_output=nb_output)
        self.exploration_weight = exploration_weight
        self.target = target
        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')  

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound 
        Use the mean and var of the folded distrib
        """
        m, s = self.model.predict(x)
        m_folded, s_folded = folded_normal(m - np.repeat(np.atleast_1d(self.target)[np.newaxis, :], len(m), 0), s)
        f_acqu = - m_folded + self.exploration_weight * s_folded
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        Use the mean and var of the folded distrib
        """
        raise NotImplementedError()
        m, s, dmdx, dsdx = self.model.predict_withGradients(x) 
        f_acqu = -m + self.exploration_weight * s       
        df_acqu = -dmdx + self.exploration_weight * dsdx
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