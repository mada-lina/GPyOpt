# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from .EI import AcquisitionEI
from GPyOpt.acquisitions.EI_mcmc import AcquisitionEI_MCMC
from .MPI import AcquisitionMPI
from .MPI_mcmc import AcquisitionMPI_MCMC
from .LCB import AcquisitionLCB
from .LCB_mcmc import AcquisitionLCB_MCMC
from .LP import AcquisitionLP
from .ES import AcquisitionEntropySearch
from .EI_target import AcquisitionEI_target
from .LCB_target import AcquisitionLCB_target
from .LCB_pspace import AcquisitionLCB_pspace
from .LCB_oneq import AcquisitionLCB_oneq


def select_acquisition(name):
    '''
    Acquisition selector
    '''
    if name == 'EI':
        return AcquisitionEI
    elif name == 'EI_MCMC':
        return AcquisitionEI_MCMC
    elif name == 'EI_TARGET':
        return AcquisitionEI_target
    elif name == 'LCB':
        return AcquisitionLCB
    elif name == 'LCB_MCMC':
        return AcquisitionLCB_MCMC
    elif name == 'LCB_TARGET':
        return AcquisitionLCB_target
    elif name == 'MPI':
        return AcquisitionMPI
    elif name == 'MPI_MCMC':
        return AcquisitionMPI_MCMC
    elif name == "LCB_oneq":
        return AcquisitionLCB_oneq
    elif name == 'LP':
        return AcquisitionLP
    elif name == 'ES':
        return AcquisitionEntropySearch
    else:
        raise Exception('Invalid acquisition selected.')
