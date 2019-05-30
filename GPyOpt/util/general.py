# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy.special import erfc, erf
import scipy.stats as stats
import scipy.special as special
import time
from ..core.errors import InvalidConfigError

def compute_integrated_acquisition(acquisition,x):
    '''
    Used to compute the acquisition function when samples of the hyper-parameters have been generated (used in GP_MCMC model).

    :param acquisition: acquisition function with GpyOpt model type GP_MCMC.
    :param x: location where the acquisition is evaluated.
    '''

    acqu_x = 0

    for i in range(acquisition.model.num_hmc_samples):
        acquisition.model.model.kern[:] = acquisition.model.hmc_samples[i,:]
        acqu_x += acquisition.acquisition_function(x)

    acqu_x = acqu_x/acquisition.model.num_hmc_samples
    return acqu_x

def compute_integrated_acquisition_withGradients(acquisition,x):
    '''
    Used to compute the acquisition function with gradients when samples of the hyper-parameters have been generated (used in GP_MCMC model).

    :param acquisition: acquisition function with GpyOpt model type GP_MCMC.
    :param x: location where the acquisition is evaluated.
    '''

    acqu_x = 0
    d_acqu_x = 0

    for i in range(acquisition.model.num_hmc_samples):
        acquisition.model.model.kern[:] = acquisition.model.hmc_samples[i,:]
        acqu_x_sample, d_acqu_x_sample = acquisition.acquisition_function_withGradients(x)
        acqu_x += acqu_x_sample
        d_acqu_x += d_acqu_x_sample

    acqu_x = acqu_x/acquisition.model.num_hmc_samples
    d_acqu_x = d_acqu_x/acquisition.model.num_hmc_samples

    return acqu_x, d_acqu_x


def best_guess(f,X):
    '''
    Gets the best current guess from a vector.
    :param f: function to evaluate.
    :param X: locations.
    '''
    n = X.shape[0]
    xbest = np.zeros(n)
    for i in range(n):
        ff = f(X[0:(i+1)])
        xbest[i] = ff[np.argmin(ff)]
    return xbest


def samples_multidimensional_uniform(bounds,num_data):
    '''
    Generates a multidimensional grid uniformly distributed.
    :param bounds: tuple defining the box constraints.
    :num_data: number of data points to generate.

    '''
    dim = len(bounds)
    Z_rand = np.zeros(shape=(num_data,dim))
    for k in range(0,dim): Z_rand[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)
    return Z_rand


def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x

def get_moments(model,x):
    '''
    Moments (mean and sdev.) of a GP model at x

    '''
    input_dim = model.X.shape[1]
    x = reshape(x,input_dim)
    fmin = min(model.predict(model.X)[0])
    m, v = model.predict(x)
    s = np.sqrt(np.clip(v, 0, np.inf))
    return (m,s, fmin)

def get_d_moments(model,x):
    '''
    Gradients with respect to x of the moments (mean and sdev.) of the GP
    :param model: GPy model.
    :param x: location where the gradients are evaluated.
    '''
    input_dim = model.input_dim
    x = reshape(x,input_dim)
    _, v = model.predict(x)
    dmdx, dvdx = model.predictive_gradients(x)
    dmdx = dmdx[:,:,0]
    dsdx = dvdx / (2*np.sqrt(v))
    return (dmdx, dsdx)


def get_quantiles(acquisition_par, fmin, m, s):
    '''
    Quantiles of the Gaussian distribution useful to determine the acquisition function values
    :param acquisition_par: parameter of the acquisition function
    :param fmin: current minimum.
    :param m: vector of means.
    :param s: vector of standard deviations.
    '''
    if isinstance(s, np.ndarray):
        s[s<1e-10] = 1e-10
    elif s< 1e-10:
        s = 1e-10
    u = (fmin - m - acquisition_par)/s
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return (phi, Phi, u)


def best_value(Y,sign=1):
    '''
    Returns a vector whose components i are the minimum (default) or maximum of Y[:i]
    '''
    n = Y.shape[0]
    Y_best = np.ones(n)
    for i in range(n):
        if sign == 1:
            Y_best[i]=Y[:(i+1)].min()
        else:
            Y_best[i]=Y[:(i+1)].max()
    return Y_best

def spawn(f):
    '''
    Function for parallel evaluation of the acquisition function
    '''
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun


def evaluate_function(f,X):
    '''
    Returns the evaluation of a function *f* and the time per evaluation
    '''
    num_data, dim_data = X.shape
    Y_eval = np.zeros((num_data, dim_data))
    Y_time = np.zeros((num_data, 1))
    for i in range(num_data):
        time_zero = time.time()
        Y_eval[i,:] = f(X[i,:])
        Y_time[i,:] = time.time() - time_zero
    return Y_eval, Y_time


def values_to_array(input_values):
    '''
    Transforms a values of int, float and tuples to a column vector numpy array
    '''
    if type(input_values)==tuple:
        values = np.array(input_values).reshape(-1,1)
    elif type(input_values) == np.ndarray:
        values = np.atleast_2d(input_values)
    elif type(input_values)==int or type(input_values)==float or type(np.int64):
        values = np.atleast_2d(np.array(input_values))
    else:
        print('Type to transform not recognized')
    return values


def merge_values(values1,values2):
    '''
    Merges two numpy arrays by calculating all possible combinations of rows
    '''
    array1 = values_to_array(values1)
    array2 = values_to_array(values2)

    if array1.size == 0:
        return array2
    if array2.size == 0:
        return array1

    merged_array = []
    for row_array1 in array1:
        for row_array2 in array2:
            merged_row = np.hstack((row_array1,row_array2))
            merged_array.append(merged_row)
    return np.atleast_2d(merged_array)


def normalize(Y, normalization_type='stats', target = None, return_normargs=False):
    """Normalize the array Y using statistics or its range.
    In case Y is effectively a 2d-array normalize by column

    :param Y: array that you want to normalize.
    :param normalization_type: String specifying the kind of normalization
    to use. Options are 'stats' to use mean and standard deviation,
    or 'maxmin' to use the range of function values.
    :param target
    
    :return Y_normalized: The normalized vector.
    """
    Y = np.asarray(Y, dtype=float)

    #if np.max(Y.shape) != Y.size:
    #    raise NotImplementedError('Only 1-dimensional arrays are supported.')

    # Only normalize with non null sdev (divide by zero). For only one
    # data point both std and ptp return 0.
    if normalization_type == 'stats':
        mean = Y.mean(0)
        std = Y.std(0)
        std[std<=0] = 1.
        args = {'mean': mean, 'std':std}
        if target is not None:
            target_norm = (target - mean) / std        
        mean = np.repeat(mean[np.newaxis, :], len(Y),0)
        std = np.repeat(std[np.newaxis, :], len(Y),0)
        Y_norm = (Y - mean)/std
        

    elif normalization_type == 'maxmin':
        Y_min = Y.min(0)
        Y_range = np.ptp(Y, 0)
        if target is not None:
            target_norm = target - Y_min
            target_norm[Y_range > 0] = 2 * (target_norm[Y_range > 0] / Y_range[Y_range > 0] - 0.5)
        Y_min = np.repeat(Y_min[np.newaxis, :], len(Y), 0)
        Y_range= np.repeat(Y_range[np.newaxis, :], len(Y), 0)         
        Y_norm = Y - Y_min
        Y_norm[Y_range > 0] = 2 * (Y_norm[Y_range > 0] / Y_range[Y_range > 0] - 0.5)
        args = {'min': Y_range[0], 'max': Y_range[1]}

    else:
        raise ValueError('Unknown normalization type: {}'.format(normalization_type))
    if return_normargs:
        if target is None:
            return Y_norm, args
        else:
            assert len(target) == Y.shape[1], "target has shape {} while Y is {}".format(np.shape(target), np.shape(Y))
            return Y_norm, target_norm, args
    else:
        if target is None:
            return Y_norm
        else:
            assert len(target) == Y.shape[1], "target has shape {} while Y is {}".format(np.shape(target), np.shape(Y))
            return Y_norm, target_norm

def invert_normalize(Y, normalization_type='stats', norm_args={}, target = None):
    """Normalize the array Y using statistics or its range.
    In case Y is effectively a 2d-array normalize by column

    :param Y: array that you want to normalize.
    :param normalization_type: String specifying the kind of normalization
    to use. Options are 'stats' to use mean and standard deviation,
    or 'maxmin' to use the range of function values.
    :param target
    
    :return Y_normalized: The normalized vector.
    """
    Y = np.asarray(Y, dtype=float)

    #if np.max(Y.shape) != Y.size:
    #    raise NotImplementedError('Only 1-dimensional arrays are supported.')

    # Only normalize with non null sdev (divide by zero). For only one
    # data point both std and ptp return 0.
    if normalization_type == 'stats':
        mean = norm_args['mean']
        std = norm_args['std']
        Y_invert = Y * std + mean

    else:
        raise ValueError('Unknown normalization type: {}'.format(normalization_type))

    return Y_invert
    

def folded_normal(m, s):
    """ From a given RV X ~ N(m, s^2) get the mean and standard deviation 
    of the distribution of |X|
    
    """
    mbys = m/s
    dev = 0.5 * mbys * mbys 
    m_folded = s * np.sqrt(2 / np.pi) * np.exp(- dev) + m * (1 - 2 * stats.norm.cdf(-mbys))
    v_folded = m * m + s * s - m_folded * m_folded    
    return m_folded, np.sqrt(v_folded)
    

def change_of_var_Phi(m, s):
    """ from arrays of mean and std of NORMAL variables X, produce the mean and std of
    Phi(X) where Phi is the normal cumulative distribution
    E[Phi(X)] = Phi(m/sqrt(1 + s^2))
    Var[Phi(X)] = E - E^2 - T (m/sqrt(1 + s^2), m/sqrt(1 + 2*s^2))
    """
    a = m / np.sqrt(1 + np.square(s))
    h = 1 / np.sqrt(1 + 2 * np.square(s))
    m_cdf = stats.norm.cdf(a)
    s_cdf = np.sqrt(m_cdf - np.square(m_cdf) - 2 * special.owens_t(a, h))
    return m_cdf, s_cdf

def change_of_var_Phi_withGradients(m, s, dmdx, dsdx):
    """ from arrays of mean and std of NORMAL variables X, produce the mean and std of
    Phi(X) where Phi is the normal cumulative distribution
    E[Phi(X)] = Phi(m/sqrt(1 + s^2))
    Var[Phi(X)] = E - E^2 - T (m/sqrt(1 + s^2), m/sqrt(1 + 2*s^2))
    RETURNS VARIANCE (and not STD)
    """
    assert np.ndim(dmdx) == np.ndim(m) + 1, 'dim of m: {}, dim of dm: {}'.formate(np.ndim(m), np.ndim(dmdx))
    assert np.ndim(dsdx) == np.ndim(s) + 1, 'dim of s: {}, dim of ds: {}'.formate(np.ndim(s), np.ndim(dsdx))
    v = np.square(s)
    a = m / np.sqrt(1 + v)
    h = 1 / np.sqrt(1 + 2 * v)
    mcdf = stats.norm.cdf(a)
    vcdf = mcdf - np.square(mcdf) - 2 * special.owens_t(a, h)
    # Carefull 
    
    dadx = dmdx / np.sqrt(1+v[:,:, None]) - a[:,:, None] * s[:,:, None] * dsdx / (1+v[:,:, None])
    #dhdx = - 2 * s * dsdx * h / (1+2*v)
    A = np.exp(-np.square(a[:,:, None])/2)
    dmcdfdx = 1/np.sqrt(2*np.pi)* A * dadx
    dTda = - A /(2*np.sqrt(2* np.pi)) * erf(a*h/np.sqrt(2))[:,:, None]
    dTdhdhdx = - (np.exp(-np.square(m*h))[:,:, None] * h[:,:, None] * s[:,:, None] * dsdx) /(2*np.pi *(1+v[:,:, None]))
    
    dTdx = dTda * dadx  + dTdhdhdx
    dvcdfdx = dmcdfdx - 2 * mcdf[:,:, None] * dmcdfdx - 2 * dTdx 

    return mcdf, vcdf, dmcdfdx, dvcdfdx


def product_indep(m_X, s_X, m_Y, s_Y):
    """ from arrays of mean and std of random variables X, produce the mean and std of
    XY
    E[XY] = E[X] * E[Y]
    Var[XY] = Var[X]Var[Y] + Var[Y] E[X]^2 + Var[X] * E[Y]^2
    """
    v_X, v_Y = np.square(s_X), np.square(s_Y)
    m_prod = m_X * m_Y
    s_prod = np.sqrt(v_X * v_Y + v_X * np.square(m_Y) + v_Y * np.square(m_X)) 
    return m_prod, s_prod

def sum_indep(m_X, s_X, m_Y, s_Y):
    """ from arrays of mean and std of random variables X, produce the mean and std of
    XY
    E[XY] = E[X] * E[Y]
    Var[XY] = 
    """
    m_sum = m_X + m_Y
    s_sum = np.sqrt(np.square(s_X) + np.square(s_Y))
    return m_sum, s_sum