# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import change_of_var_Phi
import numpy as np


class AcquisitionLCB_oneq(AcquisitionBase):
    """
    GP-Lower Confidence Bound acquisition function Customized for one qubit
    F_1q = 1/2(1 + <x>_tgt.<x>+ <x>_tgt.<x>+ <x>_tgt.<z>) 
         = 1/2(1 + <x>_tgt (2.p_x - 1) + <z>_tgt (2.p_z - 1) + <y>_tgt (2.p_y - 1))
         = 1/2(1 - <x>_tgt - <y>_tgt - <z>_tgt) + <x>_tgt.px + <y>_tgt.py + <z>_tgt.pz
    
     -- target is a list of [px, py, pz]
    """

    analytical_gradient_prediction = False 

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2, target = None, nb_output=1, oneq_type = 'full'):
        self.optimizer = optimizer
        super(AcquisitionLCB_oneq, self).__init__(model, space, optimizer, nb_output=nb_output)
        self.exploration_weight = exploration_weight
        self.target = np.array(target) 
        self.tgt_p = np.array(target)
        self.tgt_sigmas = 2 * self.tgt_p - 1
        self.oneq_type = oneq_type
        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')  


    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound 
        Use the mean and var of the folded distrib
        """
        m, s = self.model.predict(x) #mean and std of the 
        m_p, s_p = change_of_var_Phi(m, s)
        m_sigmas, s_sigmas = 2 * m_p - 1, 2 * s_p

        if(self.oneq_type == 'full'):
            m_acq = 1/2 * (1 + np.sum(m_sigmas * self.tgt_sigmas, 1))
            s_acq = 1/2 * np.sqrt(np.sum(np.square(s_sigmas) * np.square(self.tgt_sigmas), 1))
        else:
            raise NotImplementedError()

        f_acqu = m_acq + self.exploration_weight * s_acq
        return f_acqu[:, np.newaxis]

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        Use the mean and var of the folded distrib
        """
        raise NotImplementedError()

    def _compute_acq_novar(self, x):
        """
        Computes the acquisition function without the uncertainty part i.e. the expected value of the fom
        """
        m, s = self.model.predict(x) #mean and std of the 
        m_p, s_p = change_of_var_Phi(m, s)
        m_sigmas = 2 * m_p - 1
        if(self.oneq_type == 'full'):
            m_acq = 1/2 * (1 + np.sum(m_sigmas * self.tgt_sigmas, 1))
        else:
            raise NotImplementedError()
        return -m_acq[:, np.newaxis]
