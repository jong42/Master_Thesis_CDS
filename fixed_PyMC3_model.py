# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de), Jonas Gütter (jonas.aaron.guetter@uni-jena.de)

from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core import data_operations as data_op
from mb_modelbase.models_core import data_aggregation as data_aggr
import pymc3 as pm
import numpy as np
import pandas as pd
from mb_modelbase.models_core.empirical_model import EmpiricalModel
from mb_modelbase.models_core import data_operations as data_op

class FixedProbabilisticModel(Model):
    """
    A Bayesian model is fitted here. TODO: Describe this specific model
    TODO: For now, the model is fixed even for particular data. Make that more flexible
    """

    def __init__(self, name):
        super().__init__(name)
        self._model_params = None

    def _set_data(self, df, drop_silently, **kwargs):
        self._set_data_mixed(df, drop_silently)
        return ()

    def _fit(self):
        # Specify model
        basic_model = pm.Model()
        with basic_model:
            # describe prior distributions of model parameters.
            alpha = pm.Normal('alpha', mu=0, sd=10)
            beta = pm.Normal('beta', mu=0, sd=10, shape=2)
            sigma = pm.HalfNormal('sigma', sd=1)
            # specify model for the output parameter.
            mu = alpha + beta[0] * self.data['X1'] + beta[1] * self.data['X2']
            # likelihood of the observations. Observed stochastic variable
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=self.data['Y'])
            # draw samples from the posterior
            trace = pm.sample(500)
        # getting the means of the parameters from the samples
        parameter_means = pm.summary(trace).round(2).iloc[:,0]
        self._model_params = parameter_means
        return ()

    def _marginalizeout(self, keep, remove):
        # Specify model
        basic_model = pm.Model()
        with basic_model:
            # describe prior distributions of model parameters.
            alpha = pm.Normal('alpha', mu=0, sd=10)
            beta = pm.Normal('beta', mu=0, sd=10, shape=2)
            sigma = pm.HalfNormal('sigma', sd=1)
            # specify model for the output parameter.
            mu = alpha + beta[0] * self.data['X1'] + beta[1] * self.data['X2']
            # likelihood of the observations. Observed stochastic variable
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=self.data['Y'])
            # sample only over the variables in keep
            for index,name in enumerate(keep):
                keep[index] = eval(name)
            #keep = [eval(name) for name in keep] WHY DOES THIS NOT WORK? --> scope
            sampling_properties = pm.backends.NDArray(vars=keep)
            trace = pm.sample(500,trace=sampling_properties)
        # getting the means of the parameters from the samples
        parameter_means = pm.summary(trace).round(2).iloc[:,0]
        self._model_params = parameter_means
        return ()

    def _conditionout(self, keep, remove):
        # Specify model
        basic_model = pm.Model()
        with basic_model:
            # describe prior distributions of model parameters and set variables in remove to a fixed value.
            if ('alpha' in remove):
                alpha = 1
            else:
                alpha = pm.Normal('alpha', mu=0, sd=10)
            if ('beta' in remove):
                beta = [1,1]
            else:
                beta = pm.Normal('beta', mu=0, sd=10, shape=2)
            if ('sigma' in remove):
                sigma = 1
            else:
                sigma = pm.HalfNormal('sigma', sd=1)
            # specify model for the output parameter.
            if ('mu' in remove):
                mu = 1
            else:
                mu = alpha + beta[0] * self.data['X1'] + beta[1] * self.data['X2']
            # likelihood of the observations. Observed stochastic variable
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=self.data['Y'])
            # draw samples from the posterior
            trace = pm.sample(500)
        # getting the means of the parameters from the samples
        parameter_means = pm.summary(trace).round(2).iloc[:, 0]
        self._model_params = parameter_means
        return ()

    def _density(self, x):
        # Mängel: Draw from prior oder posterior?
        #         Only 3 Parameter are selectable, keine Daten
        #         Empir

        # Specify model
        basic_model = pm.Model()
        with basic_model:
            # describe prior distributions of model parameters.
            alpha = pm.Normal('alpha', mu=0, sd=10)
            beta = pm.Normal('beta', mu=0, sd=10, shape=2)
            sigma = pm.HalfNormal('sigma', sd=1)
            # specify model for the output parameter.
            mu = alpha + beta[0] * self.data['X1'] + beta[1] * self.data['X2']
            # likelihood of the observations. Observed stochastic variable
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=self.data['Y'])
            # draw samples from the posterior
            vars = [alpha, beta, sigma]
            sampling_properties = pm.backends.NDArray(vars=vars)
            trace = pm.sample(500,trace=sampling_properties)
            # Draw samples
            samples = pd.DataFrame(columns=['alpha', 'beta', 'sigma'])
            for var in ['alpha', 'beta', 'sigma']:
                samples[var] = trace.get_values(var)
            # Calculate empirical density of point x on the samples
            density = data_op.density(samples, x)
        return (density)

    def _sample(self):
        # Specify model
        basic_model = pm.Model()
        with basic_model:
            # describe prior distributions of model parameters.
            alpha = pm.Normal('alpha', mu=0, sd=10)
            beta = pm.Normal('beta', mu=0, sd=10, shape=2)
            sigma = pm.HalfNormal('sigma', sd=1)
            # specify model for the output parameter.
            mu = alpha + beta[0] * self.data['X1'] + beta[1] * self.data['X2']
            # likelihood of the observations. Observed stochastic variable
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=self.data['Y'])
            trace = pm.sample(1)
            point = trace.point(0)
        return (point)

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy._model_params = self._model_params
        return (mycopy)


### Generate data
np.random.seed(123)
# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma
# Create dataframe
df = pd.DataFrame({'X1':X1, 'X2':X2, 'Y':Y})
###

### Set up and train model
probabilistic_model_instance = FixedProbabilisticModel('probabilistic_model_instance')
probabilistic_model_instance._set_data(df,0)
#probabilistic_model_instance._fit()
#probabilistic_model_instance._marginalizeout(keep = ['alpha','beta'], remove = ['sigma','mu','X1','X2','Y_obs'])
#print(probabilistic_model_instance.fields)
#print(probabilistic_model_instance._model_params)
probabilistic_model_instance._conditionout(keep = ['alpha', 'beta'], remove = ['sigma','X1','X2','Y_obs'])
print(probabilistic_model_instance._model_params)
probabilistic_model_instance._conditionout(keep = ['sigma','X1','X2'], remove = ['alpha','beta','Y_obs'])
print(probabilistic_model_instance._model_params)
###
print(probabilistic_model_instance._density([1,1,1]))
print(probabilistic_model_instance._sample())