# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPyOpt
import collections
import numpy as np
import time
import csv
import logging

from ..util.general import best_value, normalize
from ..util.duplicate_manager import DuplicateManager
from ..core.errors import InvalidConfigError
from ..core.task.cost import CostModel
from ..optimization.acquisition_optimizer import ContextManager

logger = logging.getLogger(__name__)

try:
    from ..plotting.plots_bo import plot_acquisition, plot_convergence
except ImportError as e:
    logger.warning("Could not import plotting module: {}".format(e))



class BO(object):
    """
    Runner of Bayesian optimization loop. This class wraps the optimization loop around the different handlers.
    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param objective: GPyOpt objective class.
    :param acquisition: GPyOpt acquisition class.
    :param evaluator: GPyOpt evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GPyOpt cost class (default, none).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param de_duplication: GPyOpt DuplicateManager class. Avoids re-evaluating the objective at previous, pending or infeasible locations (default, False).
    """


    def __init__(self, model, space, objective, acquisition, evaluator, X_init, 
            Y_init=None, cost = None, normalize_Y = True, model_update_interval = 1, 
            de_duplication = False, Y_var_init=None, Y_Nrep_init = None, hp_update_interval = 1):
        self.model = model
        self.space = space
        self.objective = objective
        self.acquisition = acquisition
        self.evaluator = evaluator
        self.normalize_Y = normalize_Y
        self.model_update_interval = model_update_interval
        self.X = X_init
        self.Y = Y_init
        self.Y_var = Y_var_init
        self.Y_Nrep_init = Y_Nrep_init
        self.cost = CostModel(cost)
        self.normalization_type = 'stats' ## not added in the API
        self.de_duplication = de_duplication
        self.model_parameters_iterations = None
        self.context = None
        self.num_acquisitions = 0
        self.hp_update_interval = hp_update_interval
        if self.hp_update_interval != 1:
            print('hyperparameters are updated only each {} iterations'.format(self.hp_update_interval))

    def suggest_next_locations(self, context = None, pending_X = None, ignored_X = None):
        """
        Run a single optimization step and return the next locations to evaluate the objective.
        Number of suggested locations equals to batch_size.

        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param pending_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet) (default, None).
        :param ignored_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again (default, None).
        """
        self.model_parameters_iterations = None
        self.num_acquisitions = 0
        self.context = context
        self._update_model(self.normalization_type)

        suggested_locations = self._compute_next_evaluations(pending_zipped_X = pending_X, ignored_zipped_X = ignored_X)

        return suggested_locations

    def run_optimization(self, max_iter = 0, max_time = np.inf,  eps = 1e-8, context = None, verbosity=False, save_models_parameters= True, report_file = None, evaluations_file = None, models_file=None):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param report_file: file to which the results of the optimization are saved (default, None).
        :param evaluations_file: file to which the evalations are saved (default, None).
        :param models_file: file to which the model parameters are saved (default, None).
        """

        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.save_models_parameters = save_models_parameters
        self.report_file = report_file
        self.evaluations_file = evaluations_file
        self.models_file = models_file
        self.model_parameters_iterations = None
        self.context = context

        # --- Check if we can save the model parameters in each iteration
        if self.save_models_parameters == True:
            if not (isinstance(self.model, GPyOpt.models.GPModel) or isinstance(self.model, GPyOpt.models.GPModel_MCMC)):
                print('Models printout after each iteration is only available for GP and GP_MCMC models')
                self.save_models_parameters = False

        # --- Setting up stop conditions
        self.eps = eps
        if  (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            if self.heteroscedastic:
                self.Y = self.Y[:, 0][:, None]
                self.Y_var = self.Y[:, 1][:, None]
                self.Y_Nrep = None
            elif self.binomial:
                self.Y = self.Y[:, 0][:, None]
                self.Y_var = None
                self.Y_Nrep = self.Y[:, 1][:, None]                
            else:
                self.Y_var = None
                self.Y_Nrep = None
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y

        #NEWFS
        if(getattr(self, '_dynamic_weights', None) == 'linear'):
            maxit = self.max_iter
            weight_init = self.acquisition.exploration_weight
            update_weights = True
            def dynamics_weight(n_iter):
                return max(0.000001, weight_init * (1 - n_iter / (maxit-1)))
        else:
            update_weights = False


        # --- Initialize time cost of the evaluations
        while (self.max_time > self.cum_time):
            # --- Update model
            try:
                self._update_model(self.normalization_type)
            except np.linalg.linalg.LinAlgError:
                break

            if (self.num_acquisitions >= self.max_iter
                    or (len(self.X) > 1 and self._distance_last_evaluations() < self.eps)):
                break

            
            if(update_weights):
                self.acquisition.exploration_weight = dynamics_weight(self.num_acquisitions)


            self.suggested_sample = self._compute_next_evaluations()

            # --- Augment X
            self.X = np.vstack((self.X,self.suggested_sample))

            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1

            if verbosity:
                print("num acquisition: {}, time elapsed: {:.2f}s".format(
                    self.num_acquisitions, self.cum_time))

        # --- Stop messages and execution time
        self._compute_results()

        # --- Print the desired result in files
        if self.report_file is not None:
            self.save_report(self.report_file)
        if self.evaluations_file is not None:
            self.save_evaluations(self.evaluations_file)
        if self.models_file is not None:
            self.save_models(self.models_file)

    def _print_convergence(self):
        """
        Prints the reason why the optimization stopped.
        """

        if self.verbosity:
            if (self.num_acquisitions == self.max_iter) and (not self.initial_iter):
                print('   ** Maximum number of iterations reached **')
                return 1
            elif (self._distance_last_evaluations() < self.eps) and (not self.initial_iter):
                print('   ** Two equal location selected **')
                return 1
            elif (self.max_time < self.cum_time) and not (self.initial_iter):
                print('   ** Evaluation time reached **')
                return 0

            if self.initial_iter:
                print('** GPyOpt Bayesian Optimization class initialized successfully **')
                self.initial_iter = False


    def evaluate_objective(self):
        """
        Evaluates the objective
        """
        self.Y_new, cost_new = self.objective.evaluate(self.suggested_sample)
        self.cost.update_cost_model(self.suggested_sample, cost_new)
        
        if self.heteroscedastic:
            assert (self.Y_new.shape[1]==2), 'In heteroscedastic mode, output of the fom should be (N, 2)'
            self.Y_var_new = self.Y_new[:, 1][:, None]
            self.Y_new = self.Y_new[:, 0][:, None]
            self.Y = np.vstack((self.Y,self.Y_new))
            self.Y_var = np.vstack((self.Y_var,self.Y_var_new))
        elif self.binomial:
            assert (self.Y_new.shape[1]==2), 'In binomial mode, output of the fom should be (N, 2)'
            self.Y_Nrep_new = self.Y_new[:, 1][:, None]
            self.Y_new = self.Y_new[:, 0][:, None]
            self.Y = np.vstack((self.Y,self.Y_new))
            self.Y_Nrep = np.vstack((self.Y_Nrep,self.Y_Nrep_new))
        else:
            self.Y = np.vstack((self.Y,self.Y_new))

    def get_best(self, update=False):
        """ return location of the best point both x, y seen and expected (i.e. based on  the 
        expected values at locations already seen)"""
        if update:   
            self._update_model()
        
        x, y = self.X, self.Y        
        y_exp = self.model.predict_with_link(x)[0]
        y_exp = self._Y_std * y_exp + self._Y_mean
        if self.binomial:
            sign = 1 if self.maximize else -1
        else:
            sign = -1 if self.maximize else 1
        best_exp = np.argmin(y_exp*sign)
        x_exp = x[best_exp]
        y_exp = y_exp[best_exp]        
        

        best_seen =  np.argmin(y*sign)
        x_seen = x[best_seen]
        y_seen = y[best_seen]
        
        return (x_seen, y_seen), (x_exp, y_exp)

    def _compute_results(self):
        """
        Computes the optimum and its value.
        """
        self.Y_best = best_value(self.Y)
        self.x_opt = self.X[np.argmin(self.Y),:]
        self.fx_opt = np.min(self.Y)

    def _distance_last_evaluations(self):
        """
        Computes the distance between the last two evaluations.
        """
        if self.X.shape[0] < 2:
            # less than 2 evaluations
            return np.inf
        return np.sqrt(np.sum((self.X[-1, :] - self.X[-2, :]) ** 2))

    def _compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None):
        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        :param pending_zipped_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet).
        :param ignored_zipped_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again.
        :return:
        """

        ## --- Update the context if any
        self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)

        ### --- Activate de_duplication
        if self.de_duplication:
            duplicate_manager = DuplicateManager(space=self.space, zipped_X=self.X, pending_zipped_X=pending_zipped_X, ignored_zipped_X=ignored_zipped_X)
        else:
            duplicate_manager = None

        ### We zip the value in case there are categorical variables
        return self.space.zip_inputs(self.evaluator.compute_batch(duplicate_manager=duplicate_manager, context_manager= self.acquisition.optimizer.context_manager))

    def _update_model(self, normalization_type='stats'):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """
        if self.num_acquisitions % self.model_update_interval == 0:
            update_hp = (self.num_acquisitions % self.hp_update_interval == 0)
            # input that goes into the model (is unziped in case there are categorical variables)
            X_inmodel = self.space.unzip_inputs(self.X)

            # Y_inmodel is the output that goes into the model
            if self.normalize_Y:
                self._Y_std = self.Y.std()
                self._Y_mean = self.Y.mean()
                Y_inmodel = normalize(self.Y, normalization_type)
            else:
                self._Y_std = 1.
                self._Y_mean = 0.
                Y_inmodel = self.Y
            if self.heteroscedastic:
                Y_var_inmodel = self.Y_var / np.square(self._Y_std)
            else:
                Y_var_inmodel = None
            self.model.updateModel(X_inmodel, Y_inmodel, None, None, Y_var = Y_var_inmodel, Y_Nrep = self.Y_Nrep, update_hp=update_hp)


    def _save_model_parameter_values(self):
        if self.model_parameters_iterations is None:
            self.model_parameters_iterations = self.model.get_model_parameters()
        else:
            self.model_parameters_iterations = np.vstack((self.model_parameters_iterations,self.model.get_model_parameters()))

    def plot_acquisition(self, filename=None, label_x=None, label_y=None):
        """
        Plots the model and the acquisition function.
            if self.input_dim = 1: Plots data, mean and variance in one plot and the acquisition function in another plot
            if self.input_dim = 2: as before but it separates the mean and variance of the model in two different plots
        :param filename: name of the file where the plot is saved
        :param label_x: Graph's x-axis label, if None it is renamed to 'x' (1D) or 'X1' (2D)
        :param label_y: Graph's y-axis label, if None it is renamed to 'f(x)' (1D) or 'X2' (2D)
        """
        if self.model.model is None:
            from copy import deepcopy
            model_to_plot = deepcopy(self.model)
            if self.normalize_Y:
                Y = normalize(self.Y, self.normalization_type)
            else:
                Y = self.Y
            model_to_plot.updateModel(self.X, Y, self.X, Y)
        else:
            model_to_plot = self.model

        return plot_acquisition(self.acquisition.space.get_bounds(),
                                model_to_plot.model.X.shape[1],
                                model_to_plot.model,
                                model_to_plot.model.X,
                                model_to_plot.model.Y,
                                self.acquisition.acquisition_function,
                                self.suggest_next_locations(),
                                filename,
                                label_x,
                                label_y)

    def plot_convergence(self,filename=None):
        """
        Makes twp plots to evaluate the convergence of the model:
            plot 1: Iterations vs. distance between consecutive selected x's
            plot 2: Iterations vs. the mean of the current model in the selected sample.
        :param filename: name of the file where the plot is saved
        """
        return plot_convergence(self.X,self.Y_best,filename)

    def get_evaluations(self):
        return self.X.copy(), self.Y.copy()

    def save_report(self, report_file= None):
        """
        Saves a report with the main results of the optimization.

        :param report_file: name of the file in which the results of the optimization are saved.
        """

        with open(report_file,'w') as file:
            import GPyOpt
            import time

            file.write('-----------------------------' + ' GPyOpt Report file ' + '-----------------------------------\n')
            file.write('GPyOpt Version ' + str(GPyOpt.__version__) + '\n')
            file.write('Date and time:               ' + time.strftime("%c")+'\n')
            if self.num_acquisitions==self.max_iter:
                file.write('Optimization completed:      ' +'YES, ' + str(self.X.shape[0]).strip('[]') + ' samples collected.\n')
                file.write('Number initial samples:      ' + str(self.initial_design_numdata) +' \n')
            else:
                file.write('Optimization completed:      ' +'NO,' + str(self.X.shape[0]).strip('[]') + ' samples collected.\n')
                file.write('Number initial samples:      ' + str(self.initial_design_numdata) +' \n')

            file.write('Tolerance:                   ' + str(self.eps) + '.\n')
            file.write('Optimization time:           ' + str(self.cum_time).strip('[]') +' seconds.\n')

            file.write('\n')
            file.write('--------------------------------' + ' Problem set up ' + '------------------------------------\n')
            file.write('Problem name:                ' + self.objective_name +'\n')
            file.write('Problem dimension:           ' + str(self.space.dimensionality) +'\n')
            file.write('Number continuous variables  ' + str(len(self.space.get_continuous_dims()) ) +'\n')
            file.write('Number discrete variables    ' + str(len(self.space.get_discrete_dims())) +'\n')
            file.write('Number bandits               ' + str(self.space.get_bandit().shape[0]) +'\n')
            file.write('Noiseless evaluations:       ' + str(self.exact_feval) +'\n')
            file.write('Cost used:                   ' + self.cost.cost_type +'\n')
            file.write('Constraints:                  ' + str(self.constraints==True) +'\n')

            file.write('\n')
            file.write('------------------------------' + ' Optimization set up ' + '---------------------------------\n')
            file.write('Normalized outputs:          ' + str(self.normalize_Y) + '\n')
            file.write('Model type:                  ' + str(self.model_type).strip('[]') + '\n')
            file.write('Model update interval:       ' + str(self.model_update_interval) + '\n')
            file.write('Acquisition type:            ' + str(self.acquisition_type).strip('[]') + '\n')
            file.write('Acquisition optimizer:       ' + str(self.acquisition_optimizer.optimizer_name).strip('[]') + '\n')

            file.write('Acquisition type:            ' + str(self.acquisition_type).strip('[]') + '\n')
            if hasattr(self, 'acquisition_optimizer') and hasattr(self.acquisition_optimizer, 'optimizer_name'):
                file.write('Acquisition optimizer:       ' + str(self.acquisition_optimizer.optimizer_name).strip('[]') + '\n')
            else:
                file.write('Acquisition optimizer:       None\n')
            file.write('Evaluator type (batch size): ' + str(self.evaluator_type).strip('[]') + ' (' + str(self.batch_size) + ')' + '\n')
            file.write('Cores used:                  ' + str(self.num_cores) + '\n')

            file.write('\n')
            file.write('---------------------------------' + ' Summary ' + '------------------------------------------\n')
            file.write('Value at minimum:            ' + str(min(self.Y)).strip('[]') +'\n')
            file.write('Best found minimum location: ' + str(self.X[np.argmin(self.Y),:]).strip('[]') +'\n')

            file.write('----------------------------------------------------------------------------------------------\n')
            file.close()

    def _write_csv(self, filename, data):
        with open(filename, 'w') as csv_file:
           writer = csv.writer(csv_file, delimiter='\t')
           writer.writerows(data)

    def save_evaluations(self, evaluations_file = None):
        """
        Saves  evaluations at each iteration of the optimization

        :param evaluations_file: name of the file in which the results are saved.
        """
        iterations = np.array(range(1, self.Y.shape[0] + 1))[:, None]
        results = np.hstack((iterations, self.Y, self.X))
        header = ['Iteration', 'Y'] + ['var_' + str(k) for k in range(1, self.X.shape[1] + 1)]

        data = [header] + results.tolist()
        self._write_csv(evaluations_file, data)

    def save_models(self, models_file):
        """
        Saves model parameters at each iteration of the optimization

        :param models_file: name of the file or a file buffer, in which the results are saved.
        """
        if self.model_parameters_iterations is None:
            raise ValueError("No iterations have been carried out yet and hence no iterations of the BO can be saved")

        iterations = np.array(range(1,self.model_parameters_iterations.shape[0]+1))[:,None]
        results = np.hstack((iterations,self.model_parameters_iterations))
        header = ['Iteration'] + self.model.get_model_parameters_names()

        data = [header] + results.tolist()
        self._write_csv(models_file, data)
