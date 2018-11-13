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
        self._model_structure = None
        self._model_params = None

    def _set_data(self, df, drop_silently, **kwargs):
        # TODO: Is there a better place for decribing the whole model?
        self._set_data_mixed(df, drop_silently)
        basic_model = pm.Model()
        with basic_model:
            # describe prior distributions of model parameters.
            alpha = pm.Normal('alpha', mu=0, sd=10)
            beta1 = pm.Normal('beta1', mu=0, sd=10)
            beta2 = pm.Normal('beta2', mu=0, sd=10)
            sigma = pm.HalfNormal('sigma', sd=1)
            # specify model for the output parameter.
            mu = alpha + beta1 * self.data['X1'] + beta2 * self.data['X2']
            # likelihood of the observations. Observed stochastic variable
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=self.data['Y'])
        self._model_structure = basic_model
        return ()

    def _fit(self):
        # Specify model
        basic_model = self._model_structure
        with basic_model:
            # draw samples from the posterior
            trace = pm.sample(500)
        # getting the means of the parameters from the samples
        parameter_means = pm.summary(trace).round(2).iloc[:,0]
        self._model_params = parameter_means
        return ()

    def _marginalizeout(self, keep, remove):
        # Set data to comlete dataset for this method. change it back at the end of the method
        data = self.data
        self._set_data(df, 0)
        # Specify model
        basic_model = self._model_structure
        with basic_model:
            trace = pm.sample(500)
        # Get the values of the parameter samples
        joint_sample = np.zeros(shape=(2000,8))
        for i,var in enumerate(basic_model.unobserved_RVs):
            joint_sample[:,i] = trace.get_values(var)
        # Generate the data samples
        X1_joint_sample = np.random.randn(2000)
        X2_joint_sample = np.random.randn(2000) * 0.2
        joint_sample[:,5] = X1_joint_sample
        joint_sample[:,6] = X2_joint_sample
        mu_joint_sample = joint_sample[:,0] + np.multiply(joint_sample[:,1], joint_sample[:,5].T) + np.multiply(joint_sample[:,2], joint_sample[:,6].T)
        Y_joint_sample = np.random.normal(loc=mu_joint_sample, scale=joint_sample[:,4])
        joint_sample[:,7] = Y_joint_sample
         # Get the means of the the sampled dimensions
        joint_sample_means = []
        for col in range(0,joint_sample.shape[1]):
            joint_sample_means = np.append(joint_sample_means,np.mean(joint_sample[:,col]))
        # write only variables from keep to the model parameters
        all_vars = {'alpha':0,'beta1':1,'beta2':2,'sigma_log':3,'sigma':4,'X1':5,'X2':6,'Y':7}
        self._model_params = [joint_sample_means[index] for index in [all_vars[name] for name in keep]]
        # change data back to previous selections
        self.data = data
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
            if ('beta1' in remove):
                beta1 = 1
            else:
                beta1 = pm.Normal('beta1', mu=0, sd=10)
            if ('beta2' in remove):
                beta2 = 1
            else:
                beta2 = pm.Normal('beta2', mu=0, sd=10)
            if ('sigma' in remove):
                sigma = 1
            else:
                sigma = pm.HalfNormal('sigma', sd=1)
            # specify model for the output parameter.
            if ('mu' in remove):
                mu = 1
            else:
                mu = alpha + beta1 * self.data['X1'] + beta2 * self.data['X2']
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
        #         Empirical density

        # Specify model
        basic_model = self._model_structure
        with basic_model:
            vars = basic_model.unobserved_RVs
            sampling_properties = pm.backends.NDArray(vars=vars)
            trace = pm.sample(500,trace=sampling_properties)
            # Draw samples
            samples = pd.DataFrame(columns=vars)
            for var in vars:
                samples[var] = trace.get_values(var)
            # Calculate empirical density of point x on the samples
            density = data_op.density(samples, x)
        return (density)

    def _sample(self):
        # Specify model
        basic_model = self._model_structure
        with basic_model:
            trace = pm.sample(1)
            point = trace.point(0)
        return (point)

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy._model_params = self._model_params
        return (mycopy)


### Generate data
#np.random.seed(123)
# True parameter values
#alpha, sigma = 1, 1
#beta = [1, 2.5]

# Size of dataset
#size = 100

# Predictor variable
#X1 = np.random.randn(size)
#X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
#Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma
# Create dataframe
#df = pd.DataFrame({'X1':X1, 'X2':X2, 'Y':Y})
###

df = pd.read_csv('fixed_PyMC3_example_data.csv')

### Set up and train model
probabilistic_model_instance = FixedProbabilisticModel('probabilistic_model_instance')
probabilistic_model_instance._set_data(df,0)
#probabilistic_model_instance._fit()
probabilistic_model_instance._marginalizeout(keep = ['X1'], remove = ['alpha','beta1','beta2','sigma','mu','X1','Y_obs'])
probabilistic_model_instance._marginalizeout(keep = ['alpha'], remove = ['alpha','beta','sigma','mu','X1','Y_obs'])
print(probabilistic_model_instance.fields)
print(probabilistic_model_instance._model_params)
probabilistic_model_instance._conditionout(keep = ['alpha', 'beta1'], remove = ['beta2','sigma','X1','X2','Y_obs'])
print(probabilistic_model_instance._model_params)
probabilistic_model_instance._conditionout(keep = ['sigma','X1','X2'], remove = ['alpha','beta1','beta2','Y_obs'])
print(probabilistic_model_instance._model_params)
###
print(probabilistic_model_instance._density([1,1,1]))
print(probabilistic_model_instance._sample())