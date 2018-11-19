# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de), Jonas Gütter (jonas.aaron.guetter@uni-jena.de)

from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core import data_operations as data_op
from mb_modelbase.models_core import data_aggregation as data_aggr
import pymc3 as pm
import numpy as np
import pandas as pd
from mb_modelbase.models_core.empirical_model import EmpiricalModel
from mb_modelbase.models_core import data_operations as data_op
from sklearn.neighbors.kde import KernelDensity

class FixedProbabilisticModel(Model):
    """
    A Bayesian model is fitted here. TODO: Describe this specific model
    TODO: For now, the model is fixed even for particular data. Make that more flexible
    """

    def __init__(self, name):
        super().__init__(name)


    def _set_data(self, df, drop_silently, **kwargs):
        self._set_data_mixed(df, drop_silently)
        self._update_all_field_derivatives()
        return ()

    def _fit(self,model):
        with model:
            # Draw samples
            trace = pm.sample(500)
            self.samples = {}
            for varname in trace.varnames:
                self.samples[varname] = trace[varname]
        return ()

# Achtung: _marginalizeout is currently only for parameters possible,
# whereas _conditionout is only for data points possible

    def _marginalizeout(self, keep, remove):
        # Remove all variables in remove
        for varname in remove:
            self.samples[varname] = None
        return ()

    def _conditionout(self, keep, remove):
        names = remove
        fields = self.fields if names is None else self.byname(names)
        cond_domains = [field['domain'] for field in fields]
        return (cond_domains)

    # def _density(self, x):
    #
    #     X = np.array([self._joint_samples_alpha,self._joint_samples_beta1,
    #                   self._joint_samples_beta2, self._joint_samples_sigma,
    #                   self._joint_samples_X1, self._joint_samples_X2,
    #                   self._joint_samples_mu]).T
    #     kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
    #     x = np.reshape(x,(1,len(x)))
    #     logdensity = kde.score_samples(x)
    #     return (np.exp(logdensity))

    # def _sample(self):
    #     # Specify model
    #     basic_model = self._model_structure
    #     with basic_model:
    #         trace = pm.sample(1)
    #         point = trace.point(0)
    #     return (point)
    #
    # def copy(self, name=None):
    #     mycopy = self._defaultcopy(name)
    #     mycopy._model_structure = self._model_structure
    #     mycopy._joint_samples_alpha = self._joint_samples_alpha
    #     mycopy._joint_samples_beta1 = self._joint_samples_beta1
    #     mycopy._joint_samples_beta2 = self._joint_samples_beta2
    #     mycopy._joint_samples_sigma = self._joint_samples_sigma
    #     mycopy._joint_samples_X1 = self._joint_samples_X1
    #     mycopy._joint_samples_X2 = self._joint_samples_X2
    #     mycopy._joint_samples_mu = self._joint_samples_mu
    #     mycopy.joint_samples_Y = self._joint_samples_Y
    #
    #     return (mycopy)
    #
    #
    #

data = pd.read_csv('fixed_PyMC3_example_data.csv')

basic_model = pm.Model()
with basic_model:
    # describe prior distributions of model parameters.
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)
    # specify model for the output parameter.
    mu = alpha + beta1 * data['X1'] + beta2 * data['X2']
    # likelihood of the observations. Observed stochastic variable
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed= data['Y'])

mymodel = FixedProbabilisticModel('mymodel')
mymodel._set_data(data,1)
mymodel._fit(basic_model)
mymodel._marginalizeout(['hurz'],['sigma'])
mymodel._conditionout(['hurz'],['X1'])