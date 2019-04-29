# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import change_of_var_Phi,change_of_var_Phi_withGradients
from ..models import gpmodel
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

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2, target = None, nb_output=1, acq_nbq = 1):
        self.optimizer = optimizer
        super(AcquisitionLCB_oneq, self).__init__(model, space, optimizer, nb_output=nb_output)
        self.exploration_weight = exploration_weight
        self.target = np.array(target) 
        self.tgt_p = np.array(target)
        self.tgt_sigmas = 2 * self.tgt_p - 1
        self.acq_nbq = acq_nbq
        self.coeff_dim = 1/(2**acq_nbq) 
        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')  


    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound 
        Use the mean and var of the folded distrib
        """
        m, s = self.model.predict(x) #mean and std of the
        if type(self.model) == gpmodel.GPModelCustomLik:
            m_p, s_p = change_of_var_Phi(m, s)
        else:
            m_p, s_p = m, s
        m_sigmas, s_sigmas = 2 * m_p - 1, 2 * s_p
        m_acq = self.coeff_dim * (1 + np.sum(m_sigmas * self.tgt_sigmas, 1))
        s_acq = self.coeff_dim * np.sqrt(np.sum(np.square(s_sigmas * self.tgt_sigmas), 1))
        f_acqu = m_acq + self.exploration_weight * s_acq
        return f_acqu[:, np.newaxis]

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        Use the mean and var of the folded distrib
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x) 
        if type(self.model) == gpmodel.GPModelCustomLik:
            mp, vp, dmpdx, dvpdx = change_of_var_Phi_withGradients(m, s, dmdx, dsdx)
        else:
            mp, vp, dmpdx, dvpdx = m, np.square(s), dmdx, 2 * s * dsdx
        msigma, vsigma, dmsigmadx, dvsigmadx = (2*mp -1), 4*vp, 2*dmpdx, 4*dvpdx
        macq = self.coeff_dim * (1 + np.dot(msigma, self.tgt_sigmas))
        sacq = self.coeff_dim * np.sqrt(np.dot(vsigma, np.square(self.tgt_sigmas)))
        
        dmacqdx = self.coeff_dim * np.einsum('j,ijk', self.tgt_sigmas, dmsigmadx)
        dsacqdx = self.coeff_dim**2 * np.einsum('j,ijk', np.square(self.tgt_sigmas), dvsigmadx)/(2*sacq)
        #dsacqdx =  np.einsum('j,ijk', np.square(self.coeff_dim * self.tgt_sigmas), dvsigmadx * vsigma[:,:, np.newaxis]) /sacq
        f_acqu = macq + self.exploration_weight * sacq       
        df_acqu = dmacqdx + self.exploration_weight * dsacqdx
        return f_acqu[:, np.newaxis], df_acqu[:, np.newaxis]
        
#        eps = 1e-6
#        x_eps = x + eps * np.ones(np.shape(x))
#        m_eps, s_eps, dmdx_eps, dsdx_eps = self.model.predict_withGradients(x_eps)
#        mp_eps, vp_eps, dmpdx_eps, dvpdx_eps = change_of_var_Phi_withGradients(m_eps, s_eps, dmdx_eps, dsdx_eps)
#        (m_eps - m) /eps
#        np.sum(dmdx, 1)
#        (mp_eps - mp) /eps
#        np.sum(dmpdx, 2)
#        (vp_eps - vp) /eps
#        np.sum(dvpdx, 2)
#        msigma_eps, vsigma_eps, dmsigmadx_eps, dvsigmadx_eps = (2*mp_eps -1), 4*vp_eps, 2*dmpdx_eps, 4*dvpdx_eps
#        (msigma_eps - msigma)/eps
#        np.sum(dmsigmadx, 2)
#        macq_eps = self.coeff_dim * (1 + np.dot(msigma_eps, self.tgt_sigmas))
#        sacq_eps = self.coeff_dim * np.sqrt(np.dot(vsigma_eps, np.square(self.tgt_sigmas)))
#        dmacqdx_eps = self.coeff_dim * np.einsum('j,ijk', self.tgt_sigmas, dmsigmadx_eps)
#        dsacqdx_eps = self.coeff_dim * np.einsum('j,ijk', np.square(self.tgt_sigmas), dvsigmadx_eps)/(2*sacq_eps)
#





    def _compute_acq_novar(self, x):
        """
        Computes the acquisition function without the uncertainty part i.e. the expected value of the fom
        """
        m, s = self.model.predict(x) #mean and std of the 
        m_p, s_p = change_of_var_Phi(m, s)
        m_sigmas = 2 * m_p - 1
        m_acq = self.coeff_dim * (1 + np.sum(m_sigmas * self.tgt_sigmas, 1))
        return -m_acq[:, np.newaxis]

    def _compute_acq_splitted(self, x):
        """
        Computes the two parts (expected value, std) used in the acqu function 
        """
        m, s = self.model.predict(x) #mean and std of the
        if type(self.model) == gpmodel.GPModelCustomLik:
            m_p, s_p = change_of_var_Phi(m, s)
        else:
            m_p, s_p = m, s
        m_sigmas, s_sigmas = 2 * m_p - 1, 2 * s_p
        m_acq = self.coeff_dim * (1 + np.sum(m_sigmas * self.tgt_sigmas, 1))
        s_acq = self.coeff_dim * np.sqrt(np.sum(np.square(s_sigmas * self.tgt_sigmas), 1))
        return m_acq[:,np.newaxis], s_acq[:,np.newaxis]