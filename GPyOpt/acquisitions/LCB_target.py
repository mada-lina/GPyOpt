# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import folded_normal


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

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2, target = None):
        self.optimizer = optimizer
        super(AcquisitionLCB_target, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight
        self.target = target
        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')  

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound 
        Use the mean and var of the folded distrib
        """
        raise NotImplementedError
        m, s = self.model.predict(x)
        m_folded, s_folded = folded_normal(m - self.target, s)
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

