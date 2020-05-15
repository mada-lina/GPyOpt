# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools
from scipy.stats import norm
import GPy

from .base import BOModel
from ..util.general import folded_normal
from ..util import multioutput

class GPModel(BOModel):
    """
    General class for handling a Gaussian Process in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param exact_feval: whether noiseless evaluations are available. IMPORTANT to make the optimization work well in noiseless scenarios (default, False).
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum number of iterations used to optimize the parameters of the model.
    :param optimize_restarts: number of restarts in the optimization.
    :param sparse: whether to use a sparse GP (default, False). This is useful when many observations are available.
    :param num_inducing: number of inducing points if a sparse GP is used.
    :param verbose: print out the model messages (default, False).
    :param ARD: whether ARD is used in the kernel (default, False).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """


    analytical_gradient_prediction = True  # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs', max_iters=1000, 
        optimize_restarts=5, sparse = False, num_inducing = 10,  verbose=True, ARD=False, mo=None, mean_function=None):
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.sparse = sparse
        self.num_inducing = num_inducing
        self.model = None
        self.ARD = ARD
        self.mean_function=mean_function
        self._mf = None
        
        # workaround to deal with multiple output
        if mo is not None:
            self.mo_flag = True
            self.mo_output_dim = mo['output_dim'] 
            self.mo_rank =  mo['rank']
            self.mo_missing = mo.get('missing', False)
            self.mo_kappa = mo.get('kappa')
            self.mo_kappa_fix = mo.get('kappa_fix', False)
        else:
            self.mo_flag = False
            self.mo_output_dim = 1
    @staticmethod
    def fromConfig(config):
        return GPModel(**config)

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """



        # --- define kernel
        self.input_dim = X.shape[1]
        if type(self.mean_function) == float:
            self._mf = gen_scalar_mf(self.mean_function, self.input_dim)
            self._empirical_mf = False
        elif self.mean_function == 'empirical':
            self._empirical_mf = True
        elif type(self.mean_function) == list:
            nb_output = self.mo_output_dim
            assert len(self.mean_function) == nb_output, "len mean_function does not match nb_output"
            def coreg_mf(x):
                return np.array([np.atleast_1d(self.mean_function[int(xx[-1])]) for xx in np.atleast_2d(x)])
            self._mf = gen_func_mf(coreg_mf, self.input_dim+1)
            self._empirical_mf = False
        if self.kernel is None:
            kern = GPy.kern.Matern52(self.input_dim, variance=1., ARD=self.ARD) #+ GPy.kern.Bias(self.input_dim)
        else:
            kern = self.kernel
            self.kernel = None

        noise_var = np.average(Y.var(0))*0.01 if self.noise_var is None else self.noise_var



        if not self.sparse:
            if self.mo_flag:
                self.X_ext, self.Y_ext = multioutput.extend_XY(X, Y, self.mo_output_dim)
                self.X_init = X
                coreg = GPy.kern.Coregionalize(1, output_dim=self.mo_output_dim, rank=self.mo_rank, kappa = self.mo_kappa, name='coregion')
                if self.mo_kappa_fix:
                    coreg.kappa.fix()
                kern = kern ** coreg 
                self.model = GPy.models.GPRegression(self.X_ext, self.Y_ext, kern, Y_metadata={'output_index':self.X_ext[:, -1][:,np.newaxis]},
                        mean_function=self._mf)
            else:
                self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var,mean_function=self._mf)
        
        else:
            if self.mo_flag:
                raise NotImplementedError()

            else:
                self.model = GPy.models.SparseGPRegression(X, Y, kernel=kern, num_inducing=self.num_inducing,mean_function=self._mf)

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            # --- We make sure we do not get ridiculously small residual noise variance
            self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False) #constrain_positive(warning=False)

    def updateModel(self, X_all, Y_all, X_new, Y_new, update_hp = True):
        """
        Updates the model with new observations.
        """
        
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            if self.mo_flag:
                self.X_ext, self.Y_ext = multioutput.extend_XY(X_all, Y_all, self.mo_output_dim)
                self.X_init = X_all
                self.model.set_XY(self.X_ext, self.Y_ext)
            else:
                self.model.set_XY(X_all, Y_all)

        # WARNING: Even if self.max_iters=0, the hyperparameters are bit modified...
        if self.max_iters > 0 and update_hp:
            # --- update the model maximizing the marginal likelihood.
            if self.optimize_restarts==1:
                self.model.optimize(optimizer=self.optimizer, max_iters = self.max_iters, messages=False, ipython_notebook=False)
            else:
                self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer, max_iters = self.max_iters, verbose=self.verbose)

    def _predict(self, X, full_cov, include_likelihood):
        """ Use the underlying GP model to make predictions
        In  case of multioutput return in the shape nb_X x nb_output"""
        if X.ndim == 1:
            X = X[None,:]
        X_ext = multioutput.extend_X(X, self.mo_output_dim) if(self.mo_flag) else X
        m, v = self.model.predict(X_ext, full_cov=full_cov, include_likelihood=include_likelihood)
        v = np.clip(v, 1e-10, np.inf)
        if self.mo_flag:
            m = m.reshape(len(X), self.mo_output_dim)
            v = v.reshape(len(X), self.mo_output_dim)

        return m, v


    def predict(self, X, with_noise=True):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        m, v = self._predict(X, False, with_noise)
        # We can take the square root because v is just a diagonal matrix of variances
        return m, np.sqrt(v)

    def predict_covariance(self, X, with_noise=True):
        """
        Predicts the covariance matric for points in X.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        _, v = self._predict(X, True, with_noise)
        return v

    def get_model_predict(self, include_likelihood = True):
        """ Return the prediction for the already seen location"""
        mu, v = self.model.predict(self.model.X, include_likelihood=include_likelihood)
        _, list_muv= multioutput.contract_XYs(self.model.X, [mu, v], nb_index = self.mo_output_dim)
        return list_muv[0], list_muv[1]

    def get_model_data(self):
        """ Return the data used by the model in the right format (e.g. if multioutput )"""
        X, Y = multioutput.contract_XY(self.model.X, self.model.Y, nb_index = self.mo_output_dim)
        return X, Y


    def get_fmin(self, target = None, mo_avg = False, fold = False):
        """ Returns the minimal value of the posterior (potentially altered) at locations already visited. 
        params:
            target: shift the mean of the distrib
            fold: use the folded distribution
            mo_avg return the mean of the average over all the output

        """
        mu, v = self.get_model_predict()
        if target is not None:
            assert np.size(target) == mu.shape[1]
            mu = mu - np.repeat(np.squeeze(target)[np.newaxis, :], len(mu), 0)
        if fold:
            mu, _  = folded_normal(mu, v)
        elif target is not None: 
            mu = np.abs(mu)
        if mo_avg:
            mu = np.average(mu, 1)[:, np.newaxis]
        fmin = np.squeeze(np.min(mu, 0))
        return fmin
    

    def predict_withGradients(self, X,include_likelihood = False):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        if X.ndim==1: X = X[None,:]
        X_ext = multioutput.extend_X(X, self.mo_output_dim) if self.mo_flag else X
        m, v = self.model.predict(X_ext, include_likelihood=include_likelihood)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(X_ext)
        dmdx = dmdx[:,:,0]
        dsdx = dvdx / (2*np.sqrt(v))
        if self.mo_flag:
            nb_params = X.shape[1]
            nb_X = X.shape[0]
            m = m.reshape(nb_X, self.mo_output_dim) # nbx, nbmodels
            v = v.reshape(nb_X, self.mo_output_dim) # nbx, nbmodels
            dmdx = dmdx[:,:-1].reshape(nb_X, self.mo_output_dim, nb_params) # nbx, nbmodels, nb_params
            dsdx = dsdx[:,:-1].reshape(nb_X, self.mo_output_dim, nb_params) # nbx, nbmodels, nb_params
        return m, np.sqrt(v), dmdx, dsdx


    def copy(self):
        """
        Makes a safe copy of the model.
        """
        copied_model = GPModel(kernel = self.model.kern.copy(),
                            noise_var=self.noise_var,
                            exact_feval=self.exact_feval,
                            optimizer=self.optimizer,
                            max_iters=self.max_iters,
                            optimize_restarts=self.optimize_restarts,
                            verbose=self.verbose,
                            ARD=self.ARD,
                            mean_function=self.mean_function)

        copied_model._create_model(self.model.X,self.model.Y)
        copied_model.updateModel(self.model.X,self.model.Y, None, None)
        return copied_model

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.model.parameter_names_flat().tolist()

    def get_covariance_between_points(self, x1, x2):
        """
        Given the current posterior, computes the covariance between two sets of points.
        """
        return self.model.posterior_covariance_between_points(x1, x2)


class GPModelCustomLik(BOModel):
    """
    General class for handling a Gaussian Process with custom likelihood (i.e. 
    Binomial/Bernouilli)

    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param exact_feval: whether noiseless evaluations are available. IMPORTANT to make the optimization work well in noiseless scenarios (default, False).
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum number of iterations used to optimize the parameters of the model.
    :param optimize_restarts: number of restarts in the optimization.
    :param sparse: whether to use a sparse GP (default, False). This is useful when many observations are available.
    :param num_inducing: number of inducing points if a sparse GP is used.
    :param verbose: print out the model messages (default, False).
    :param ARD: whether ARD is used in the kernel (default, False).

    .. Note as the posterior is non analytical needs to use some non-default inference method
    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """


    analytical_gradient_prediction = True  # --- Needed in all models to check is the gradients of acquisitions are computable.
    _inf_method_map = {'EP':GPy.inference.latent_function_inference.expectation_propagation.EP, 
                        'Laplace':GPy.inference.latent_function_inference.Laplace}
    _likelihood_map = {'Bernoulli':GPy.likelihoods.Bernoulli, 'Binomial':GPy.likelihoods.Binomial}

    def __init__(self, likelihood = 'Bernoulli',  inf_method = 'Laplace', gp_link=None, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs', 
                 max_iters=1000, optimize_restarts=5, sparse = False, num_inducing = 10,  
                 verbose=True, ARD=False, mo=None,mean_function=None):
        """
        Two new parameters:
            - inf_method: Inference method used to deqal with the non default likelihood
            - gp_link: link (map between f(x) and the probability distribution used to draw the 
                observations by default None implies a probit link)
        NOTES
        ----
        - include likelihood is set by default to False

        """
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.sparse = sparse
        self.num_inducing = num_inducing
        self.model = None
        self.ARD = ARD
        self.mean_function = mean_function
        self._mf = None

         # workaround to deal with multiple output
        if mo is not None:
            self.mo_flag = True
            self.mo_output_dim = mo['output_dim'] 
            self.mo_rank =  mo['rank']
            self.mo_missing = mo.get('missing', False)
            self.mo_kappa = mo.get('kappa')
            self.mo_kappa_fix = mo.get('kappa_fix', False)
        else:
            self.mo_flag = False
            self.mo_output_dim = 1
    

        split_likelihood = likelihood.split("_")
        self.likelihood = self._likelihood_map[split_likelihood[0]](gp_link = gp_link)
        if(self.mo_flag):
            self.likelihood_list = [self._likelihood_map[split_likelihood[0]](gp_link = gp_link) for o in range(self.mo_output_dim)]
        
        self.nb_obs = int(split_likelihood[1]) if(split_likelihood[0] == 'Binomial') else 1
        self.inf_meth = self._inf_method_map[inf_method]()


    def _gen_YData(self, Y):
        """
        Gen Y metadata to match the shape of Y. Used for Binomial likelihood
        """
        return {'trials':np.ones_like(Y) * self.nb_obs}


    @staticmethod
    def fromConfig(config):
        return GPModel(**config)

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        # --- dealing with mean functions (tricky)
        self.input_dim = X.shape[1]
        self._empirical_mf = False
        if type(self.mean_function) == float:
            self._mf = gen_scalar_mf(self.mean_function, self.input_dim)
        elif type(self.mean_function) is str:
            split = self.mean_function.split('_')
            if split[0] == 'empirical':
                self._empirical_mf = True
                if len(split) == 1:
                    self._min_update_mf = 0
                else:
                    self._min_update_mf = int(split[1])
                if self.mo_flag:
                    if len(X) > self._min_update_mf:
                        mean_emp = norm.ppf(np.average(Y, 0) / self.nb_obs)
                    else:
                        mean_emp = np.zeros(np.shape(Y)[1])
                    def coreg_mf(x):
                        return np.array([np.atleast_1d(mean_emp[int(xx[-1])]) for xx in np.atleast_2d(x)])
                    self._mf = gen_func_mf(coreg_mf, self.input_dim+1)          
                else:
                    if len(X) > self._min_update_mf:
                        mean_emp = norm.ppf(np.average(Y) / self.nb_obs)
                    else:
                        mean_emp = 0
                    self._mf = gen_scalar_mf(mean_emp, self.input_dim)
            else:
                raise NotImplementedError()
        elif type(self.mean_function) == list:
            nb_output = self.mo_output_dim
            assert len(self.mean_function) == nb_output, "len mean_function does not match nb_output"
            def coreg_mf(x):
                return np.array([np.atleast_1d(self.mean_function[int(xx[-1])]) for xx in np.atleast_2d(x)])
            self._mf = gen_func_mf(coreg_mf, self.input_dim+1)
            

        # --- define kernel
        if self.kernel is None:
            kern = GPy.kern.Matern52(self.input_dim, variance=1., ARD=self.ARD) #+ GPy.kern.Bias(self.input_dim)
        else:
            kern = self.kernel
            self.kernel = None

                
        # --- define model
        if not(self.exact_feval):
            noise_var = Y.var()*0.01 if self.noise_var is None else self.noise_var
            kern += GPy.kern.White(self.input_dim, noise_var)
            #kern.name = 'kerX'
            kern.white.constrain_bounded(1e-9, 1e6, warning=False) #constrain_positive(warning=False)
            
            
        if not self.sparse:
            if self.mo_flag:
                coreg = GPy.kern.Coregionalize(1, output_dim=self.mo_output_dim, rank=self.mo_rank, kappa = self.mo_kappa, name='coregion')
                if self.mo_kappa_fix:
                    coreg.kappa.fix()
                kern = kern ** coreg 
                self.X_ext, self.Y_ext = multioutput.extend_XY(X, Y, self.mo_output_dim)
                self.X_init = X
                Y_metadata= self._gen_YData(self.Y_ext)
                Y_metadata.update({'output_index':self.X_ext[:, -1][:,np.newaxis]})
                self.model = GPy.core.GP(self.X_ext, self.Y_ext, kern , inference_method=self.inf_meth, 
                                         likelihood=self.likelihood, normalizer=False, 
                                         name='CoregCustomLik', Y_metadata = Y_metadata, mean_function=self._mf)

            else:
                Y_metadata= self._gen_YData(Y)
                self.model = GPy.core.GP(X, Y, kernel=kern, inference_method=self.inf_meth, likelihood=self.likelihood, 
                                                normalizer=False, Y_metadata = Y_metadata,mean_function=self._mf)
        else:
            raise NotImplementedError()

        # --- restrict variance if exact evaluations of the objective
        #if not(self.exact_feval):
            # --- We make sure we do not get ridiculously small residual noise variance
        #    self.model.kern.white.constrain_bounded(1e-9, 1e6, warning=False) #constrain_positive(warning=False)


    def updateModel(self, X_all, Y_all, X_new, Y_new, update_hp=True):
        """
        Updates the model with new observations.
        """
        
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            if self.mo_flag:
                self.X_ext, self.Y_ext = multioutput.extend_XY(X_all, Y_all, self.mo_output_dim)
                Y_metadata = self._gen_YData(self.Y_ext)
                Y_metadata.update({'output_index':self.X_ext[:, -1][:,np.newaxis]})
                self.model.Y_metadata = Y_metadata
                self.model.set_XY(self.X_ext, self.Y_ext)
            else:
                self.model.Y_metadata = self._gen_YData(Y_all)
                self.model.set_XY(X_all, Y_all)

        # WARNING: Even if self.max_iters=0, the hyperparameters are bit modified...
        if self.max_iters > 0 and update_hp:
            # --- update the model maximizing the marginal likelihood.
            if self._empirical_mf:
                if self.mo_flag:
                    if len(X_all) >  self._min_update_mf:
                        mean_emp = norm.ppf(np.average(Y_all, 0) / self.nb_obs)
                    else:
                        mean_emp = np.zeros(np.shape(Y_all)[1])
                    def coreg_mf(x):
                        return np.array([np.atleast_1d(mean_emp[int(xx[-1])]) for xx in np.atleast_2d(x)])
                    self.model.mean_function = gen_func_mf(coreg_mf, self.input_dim+1)     
                else:
                    if len(X_all) >  self._min_update_mf:
                        mean_emp = norm.ppf(np.average(Y_all) / self.nb_obs)
                    else:
                        mean_emp = 0
                    self.model.mean_function = gen_scalar_mf(mean_emp, self.input_dim)
            if self.optimize_restarts==1:
                self.model.optimize(optimizer=self.optimizer, max_iters = self.max_iters, messages=False, ipython_notebook=False)
            else:
                self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer, max_iters = self.max_iters, verbose=self.verbose)


    def _predict(self, X, full_cov, include_likelihood):
        if X.ndim == 1:
            X = X[None,:]
        if(self.mo_flag):
            X_ext = multioutput.extend_X(X, self.mo_output_dim)
        else:
            X_ext = X
        m, v = self.model.predict(X_ext, full_cov=full_cov, include_likelihood=include_likelihood)
        v = np.clip(v, 1e-10, np.inf)
        if self.mo_flag:
            m = m.reshape(len(X), self.mo_output_dim)
            v = v.reshape(len(X), self.mo_output_dim)
        return m, v

    def predict(self, X, with_noise=False):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. 
        Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.

        Notes:
            include with_noise is now by default set to False.
            This has several implications all over the place
    
        """
        m, v = self._predict(X, False, with_noise)
        # We can take the square root because v is just a diagonal matrix of variances
        return m, np.sqrt(v)

    def predict_covariance(self, X, with_noise=True):
        """
        Predicts the covariance matric for points in X.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        _, v = self._predict(X, True, with_noise)
        return v

    def get_model_predict(self, include_likelihood = False):
        """ Return the prediction for the already seen location"""
        mu, v = self.model.predict(self.model.X, include_likelihood=include_likelihood,
                                   Y_metadata = self.model.Y_metadata)
        _, (mu, v)= multioutput.contract_XYs(self.model.X, [mu, v], nb_index = self.mo_output_dim)
        return mu, v

    def get_model_data(self):
        """ Return the data used by the model in the right format (e.g. if multioutput )"""
        X, Y = multioutput.contract_XY(self.model.X, self.model.Y, nb_index = self.mo_output_dim)
        return X, Y



    def get_fmin(self, include_likelihood = False, target = None, mo_avg = False, fold = False):
        """ Returns the minimal value of the posterior (potentially altered) at locations already visited. 
        params:
            target: shift the mean of the distrib
            fold: use the folded distribution
            mo_avg return the mean of the average over all the output

        """

        mu, v = self.get_model_predict(include_likelihood = include_likelihood)
        if target is not None:
            assert np.size(target) == mu.shape[1]
            mu = mu - np.repeat(np.squeeze(target)[np.newaxis, :], len(mu), 0)
        if fold:
            mu, _  = folded_normal(mu, np.sqrt(v))
        elif target is not None: 
            mu = np.abs(mu)
        if mo_avg:
            mu = np.average(mu, 1)[:, np.newaxis]
        fmin = np.squeeze(np.min(mu, 0))
        return fmin
    
    
    def predict_withGradients(self, X,include_likelihood = False):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        if X.ndim==1: X = X[None,:]
        X_ext = multioutput.extend_X(X, self.mo_output_dim) if self.mo_flag else X
        m, v = self.model.predict(X_ext, full_cov=False, include_likelihood=include_likelihood)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(X_ext)
        dmdx = dmdx[:,:,0]
        dsdx = dvdx / (2*np.sqrt(v))
        if self.mo_flag:
            nb_params = X.shape[1]
            nb_X = X.shape[0]
            m = m.reshape(nb_X, self.mo_output_dim) # nbx, nbmodels
            v = v.reshape(nb_X, self.mo_output_dim) # nbx, nbmodels
            dmdx = dmdx[:,:-1].reshape(nb_X, self.mo_output_dim, nb_params) # nbx, nbmodels, nb_params
            dsdx = dsdx[:,:-1].reshape(nb_X, self.mo_output_dim, nb_params) # nbx, nbmodels, nb_params
        return m, np.sqrt(v), dmdx, dsdx


    def copy(self):
        """
        Makes a safe copy of the model.
        """
        copied_model = GPModel(kernel = self.model.kern.copy(),
                            noise_var=self.noise_var,
                            exact_feval=self.exact_feval,
                            optimizer=self.optimizer,
                            max_iters=self.max_iters,
                            optimize_restarts=self.optimize_restarts,
                            verbose=self.verbose,
                            ARD=self.ARD,
                            mean_function=self.mean_function)

        copied_model._create_model(self.model.X,self.model.Y)
        copied_model.updateModel(self.model.X,self.model.Y, None, None)
        return copied_model

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.model.parameter_names_flat().tolist()

    def get_covariance_between_points(self, x1, x2):
        """
        Given the current posterior, computes the covariance between two sets of points.
        """
        return self.model.posterior_covariance_between_points(x1, x2)



class GPStacked(GPModel):
    """
    Stacked Models (cf.google Vizier paper)
    :param alpha
    :param prev previous regressor

    :other cf. GPMolde
    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.
    """
    analytical_gradient_prediction = True  # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self, prev = None, alpha=1, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs', 
        max_iters=1000, optimize_restarts=5, sparse = False, num_inducing = 10,  verbose=True, ARD=False, mean_function=None):
        self.alpha = alpha
        self.prev = prev        
        super(GPStacked,self).__init__(kernel=kernel, noise_var=noise_var, exact_feval=exact_feval, optimizer=optimizer, 
            max_iters=max_iters, optimize_restarts=optimize_restarts, sparse=sparse, num_inducing=num_inducing, 
            verbose=verbose, ARD=ARD, mean_function=mean_function)

    def _get_residuals(self, X, Y):
        residu = Y - self.prev._predict(X)
        return residu
    
    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """
        Y_res = self._get_residuals(X, Y)

        # --- define kernel
        self.input_dim = X.shape[1]
        if type(self.mean_function) == float:
            nb_output = self.mo_output_dim
            self._mf = GPy.core.Mapping(input_dim=self.input_dim, output_dim=1)
            self._mf.f = lambda x: np.array([self.mean_function for xx in np.atleast_2d(x)])
            self._mf.update_gradients = lambda a,b: 0
            self._mf.gradients_X = lambda a,b: 0
        elif type(self.mean_function) == list:
            nb_output = self.mo_output_dim
            assert len(self.mean_function) == nb_output, "len mean_function does not match nb_output"
            def coreg_mf(x):
                return np.array([np.atleast_1d(self.mean_function[int(xx[-1])]) for xx in np.atleast_2d(x)])
            self._mf = GPy.core.Mapping(input_dim=self.input_dim+1, output_dim=1)
            self._mf.f =  coreg_mf
            self._mf.update_gradients = lambda a,b: 0
            self._mf.gradients_X = lambda a,b: 0
        if self.kernel is None:
            kern = GPy.kern.Matern52(self.input_dim, variance=1., ARD=self.ARD) #+ GPy.kern.Bias(self.input_dim)
        else:
            kern = self.kernel
            self.kernel = None

        # --- define model
        noise_var = Y_res.var()*0.01 if self.noise_var is None else self.noise_var

        if not self.sparse:
            self.model = GPy.models.GPRegression(X, Y_res, kernel=kern, noise_var=noise_var, mean_function=self._mf)
        else:
            self.model = GPy.models.SparseGPRegression(X, Y_res, kernel=kern, num_inducing=self.num_inducing, mean_function=self._mf)

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            # --- We make sure we do not get ridiculously small residual noise variance
            self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False) #constrain_positive(warning=False)

    def updateModel(self, X_all, Y_all, X_new, Y_new, update_hp=True):
        """
        Updates the model with new observations.
        """
        Y_all_res = self._get_residuals(X_all, Y_all)
        if self.model is None:
            self._create_model(X_all, Y_all_res)
        else:
            self.model.set_XY(X_all, Y_all_res)

        # WARNING: Even if self.max_iters=0, the hyperparameters are bit modified...
        if self.max_iters > 0 and update_hp:
            # --- update the model maximizing the marginal likelihood.
            if self.optimize_restarts==1:
                self.model.optimize(optimizer=self.optimizer, max_iters = self.max_iters, messages=False, ipython_notebook=False)
            else:
                self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer, max_iters = self.max_iters, verbose=self.verbose)

    def _predict(self, X, full_cov, include_likelihood):
        if X.ndim == 1:
            X = X[None,:]
        m, v = self.model.predict(X, full_cov=full_cov, include_likelihood=include_likelihood)
        v = np.clip(v, 1e-10, np.inf)
        return m, v

    def predict(self, X, with_noise=True):
        """
        Predictions with the model. Returns posterior means and standard deviations at X.
        According to algo. in google Vizier

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        m, v = self._predict(X, False, with_noise)
        m_prev, v_prev = self.prev.predict(X, with_noise)
        D_new = len(self.nodel.Y)
        D_old = len(self.prev.Y)
        beta = (self.alpha * D_new)/ (self.alpha * D_new + self.alpha * D_old)
        m_new = m + m_prev 
        v_new = np.sqrt(v)**beta * np.sqrt(v_prev)**(1-beta)
        # We can take the square root because v is just a diagonal matrix of variances
        return m_new, v_new 

    def predict_covariance(self, X, with_noise=True):
        """
        Predicts the covariance matric for points in X.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        #_, v = self._predict(X, True, with_noise)
        #_, v_prev = self.prev._predict(X, True, with_noise)
        #D_new = len(self.nodel.Y)
        #D_old = len(self.prev.Y)
        #beta = (self.alpha * D_new)/ (self.alpha * D_new + self.alpha * D_old)
        # return         v**(2*beta) * v_prev**(2*(1-beta))
        raise NotImplementedError()

    def get_fmin(self, include_likelihood = False):
        """
        Returns the location where the posterior mean takes its minimal value.
        """
        return self.model.predict(self.model.X, include_likelihood=include_likelihood)[0].min()

    def get_fmin_target(self, include_likelihood = False, target = None):
        """
        Returns the location where the posterior mean takes the closest value to
        a target.
        """
        if(target is None):
            return self.get_fmin(include_likelihood)
        else:
            abs_dev = np.abs(self.model.predict(self.model.X, include_likelihood
                             =include_likelihood)[0] - target).min()
        return abs_dev

    def get_fmin_folded_normal(self, include_likelihood = False, target = None):
        """
        Returns the location of the point where the mean of |f(x) - tgt| is min
        Remarks: different from get_fmin_target 
                 mean(|f(x) - target |) != |mean(f(x)) - target|
        Use of the mean value of the folded distrib (distrib of |f(x) - target)
        """
        if(target is None):
            return self.get_fmin(include_likelihood)
        else:
            raise NotImplementedError("Is it well implemented")
            m,v = self.model.predict(self.model.X, include_likelihood=include_likelihood)
            m_folded, _ = folded_normal(m, v)
        return m_folded.min()


    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        prev is frozen ==> mean sd gradient comes only from the current model
        """
        if X.ndim==1: X = X[None,:]
        m, v = self._predict(X)
        m_prev, v_prev = self.prev.predict(X)
        D_new = len(self.nodel.Y)
        D_old = len(self.prev.Y)
        beta = (self.alpha * D_new)/ (self.alpha * D_new + self.alpha * D_old)
        m_new = m + m_prev 
        v_new = np.sqrt(v)**beta * np.sqrt(v_prev)**(1-beta)
        
        v_new = np.clip(v_new, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(X)
        dmdx = dmdx[:,:,0]
        dsdx = dvdx / (2*np.sqrt(v))

        return m_new, v_new, dmdx, dsdx

    def copy(self):
        """
        Makes a safe copy of the model.
        """
        copied_model = GPStacked(prev = self.prev, alpha = self.alpha,
                            kernel = self.model.kern.copy(),
                            noise_var=self.noise_var,
                            exact_feval=self.exact_feval,
                            optimizer=self.optimizer,
                            max_iters=self.max_iters,
                            optimize_restarts=self.optimize_restarts,
                            verbose=self.verbose,
                            ARD=self.ARD, 
                            mean_function=self.mean_function)

        copied_model._create_model(self.model.X,self.model.Y)
        copied_model.updateModel(self.model.X,self.model.Y, None, None)
        return copied_model

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.model.parameter_names_flat().tolist()

    def get_covariance_between_points(self, x1, x2):
        """
        Given the current posterior, computes the covariance between two sets of points.
        """
        return self.model.posterior_covariance_between_points(x1, x2)    


class GPModel_MCMC(BOModel):
    """
    General class for handling a Gaussian Process in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param exact_feval: whether noiseless evaluations are available. IMPORTANT to make the optimization work well in noiseless scenarios (default, False).
    :param n_samples: number of MCMC samples.
    :param n_burnin: number of samples not used.
    :param subsample_interval: sub-sample interval in the MCMC.
    :param step_size: step-size in the MCMC.
    :param leapfrog_steps: ??
    :param verbose: print out the model messages (default, False).

    .. Note:: This model does MCMC over the hyperparameters.

    """

    MCMC_sampler = True
    analytical_gradient_prediction = True # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self, kernel=None, noise_var=None, exact_feval=False, n_samples = 10, n_burnin = 100, subsample_interval = 10, 
                    step_size = 1e-1, leapfrog_steps=20, verbose=False, mean_function=None):
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.verbose = verbose
        self.n_samples = n_samples
        self.subsample_interval = subsample_interval
        self.n_burnin = n_burnin
        self.step_size = step_size
        self.leapfrog_steps = leapfrog_steps
        self.model = None
        self.mean_function = mean_function
        self._mf = None
        
    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        # --- define kernel
        self.input_dim = X.shape[1]
        if type(self.mean_function) == float:
            nb_output = self.mo_output_dim
            self._mf = GPy.core.Mapping(input_dim=self.input_dim, output_dim=1)
            self._mf.f = lambda x: np.array([self.mean_function for xx in np.atleast_2d(x)])
            self._mf.update_gradients = lambda a,b: 0
            self._mf.gradients_X = lambda a,b: 0
        elif type(self.mean_function) == list:
            nb_output = self.mo_output_dim
            assert len(self.mean_function) == nb_output, "len mean_function does not match nb_output"
            def coreg_mf(x):
                return np.array([np.atleast_1d(self.mean_function[int(xx[-1])]) for xx in np.atleast_2d(x)])
            self._mf = GPy.core.Mapping(input_dim=self.input_dim+1, output_dim=1)
            self._mf.f =  coreg_mf
            self._mf.update_gradients = lambda a,b: 0
            self._mf.gradients_X = lambda a,b: 0
        if self.kernel is None:
            kern = GPy.kern.RBF(self.input_dim, variance=1.)
        else:
            kern = self.kernel
            self.kernel = None

        # --- define model
        noise_var = Y.var()*0.01 if self.noise_var is None else self.noise_var
        self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var,mean_function=self._mf)

        # --- Define prior on the hyper-parameters for the kernel (for integrated acquisitions)
        self.model.kern.set_prior(GPy.priors.Gamma.from_EV(2.,4.))
        self.model.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(2.,4.))

        # --- Restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            self.model.Gaussian_noise.constrain_positive(warning=False)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """

        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.set_XY(X_all, Y_all)

        # update the model generating hmc samples
        self.model.optimize(max_iters = 200)
        self.model.param_array[:] = self.model.param_array * (1.+np.random.randn(self.model.param_array.size)*0.01)
        self.hmc = GPy.inference.mcmc.HMC(self.model, stepsize=self.step_size)
        ss = self.hmc.sample(num_samples=self.n_burnin + self.n_samples* self.subsample_interval, hmc_iters=self.leapfrog_steps)
        self.hmc_samples = ss[self.n_burnin::self.subsample_interval]

    def predict(self, X):
        """
        Predictions with the model for all the MCMC samples. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        """

        if X.ndim==1: X = X[None,:]
        ps = self.model.param_array.copy()
        means = []
        stds = []
        for s in self.hmc_samples:
            if self.model._fixes_ is None:
                self.model[:] = s
            else:
                self.model[self.model._fixes_] = s
            self.model._trigger_params_changed()
            m, v = self.model.predict(X)
            means.append(m)
            stds.append(np.sqrt(np.clip(v, 1e-10, np.inf)))
        self.model.param_array[:] = ps
        self.model._trigger_params_changed()
        return means, stds

    def get_fmin(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        ps = self.model.param_array.copy()
        fmins = []
        for s in self.hmc_samples:
            if self.model._fixes_ is None:
                self.model[:] = s
            else:
                self.model[self.model._fixes_] = s
            self.model._trigger_params_changed()
            fmins.append(self.model.predict(self.model.X)[0].min())
        self.model.param_array[:] = ps
        self.model._trigger_params_changed()

        return fmins


    def get_fmin_target(self, include_likelihood = False, target = None):
        """
        Returns the location where the posterior mean takes the closest value to
        a target.
        """
        raise NotImplementedError()
        
    def get_fmin_folded_normal(self, include_likelihood = False, target = None):
        """
        Returns the location of the point where the mean of |f(x) - tgt| is min
        Remarks: different from get_fmin_target 
                 mean(|f(x) - target |) != |mean(f(x)) - target|
        Use of the mean value of the folded distrib (distrib of |f(x) - target)
        """
        raise NotImplementedError()


    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X for all the MCMC samples.
        """
        if X.ndim==1: X = X[None,:]
        ps = self.model.param_array.copy()
        means = []
        stds = []
        dmdxs = []
        dsdxs = []
        for s in self.hmc_samples:
            if self.model._fixes_ is None:
                self.model[:] = s
            else:
                self.model[self.model._fixes_] = s
            self.model._trigger_params_changed()
            m, v = self.model.predict(X)
            std = np.sqrt(np.clip(v, 1e-10, np.inf))
            dmdx, dvdx = self.model.predictive_gradients(X)
            dmdx = dmdx[:,:,0]
            dsdx = dvdx / (2*std)
            means.append(m)
            stds.append(std)
            dmdxs.append(dmdx)
            dsdxs.append(dsdx)
        self.model.param_array[:] = ps
        self.model._trigger_params_changed()
        return means, stds, dmdxs, dsdxs

    def copy(self):
        """
        Makes a safe copy of the model.
        """

        copied_model = GPModel( kernel = self.model.kern.copy(),
                                noise_var= self.noise_var ,
                                exact_feval= self.exact_feval,
                                n_samples = self.n_samples,
                                n_burnin = self.n_burnin,
                                subsample_interval = self.subsample_interval,
                                step_size = self.step_size,
                                leapfrog_steps= self.leapfrog_steps,
                                verbose= self.verbose, 
                                mean_function=self.mean_function)

        copied_model._create_model(self.model.X,self.model.Y)
        copied_model.updateModel(self.model.X,self.model.Y, None, None)
        return copied_model

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.model.parameter_names()


# gen mean functions
def gen_scalar_mf(val, input_dim):
    mf = GPy.core.Mapping(input_dim=input_dim, output_dim=1)
    mf.f = lambda x: np.array([[val] for xx in np.atleast_2d(x)])
    mf.update_gradients = lambda a,b: 0
    mf.gradients_X = lambda a,b: 0
    return mf

def gen_func_mf(func, input_dim):
    mf = GPy.core.Mapping(input_dim=input_dim, output_dim=1)
    mf.f = func
    mf.update_gradients = lambda a,b: 0
    mf.gradients_X = lambda a,b: 0
    return mf