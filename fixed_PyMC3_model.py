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

    def _fit(self):
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
            # Draw samples
            trace = pm.sample(500)
        self._model_structure = basic_model
        self._joint_samples_alpha = trace.get_values('alpha')
        self._joint_samples_beta1 = trace.get_values('beta1')
        self._joint_samples_beta2 = trace.get_values('beta2')
        self._joint_samples_sigma = trace.get_values('sigma')
        self._joint_samples_X1 = np.random.randn(2000)
        self._joint_samples_X2 = np.random.randn(2000) * 0.2
        self._joint_samples_mu = self._joint_samples_alpha + np.multiply(self._joint_samples_beta1,self._joint_samples_X1.T) + np.multiply(self._joint_samples_beta2,self._joint_samples_X2.T)
        joint_samples_Y = np.random.normal(loc=self._joint_samples_mu, scale=self._joint_samples_sigma)
        return ()

    def _marginalizeout(self, keep, remove):
        # Remove all variables in remove
        if 'alpha' in remove:
            self._joint_samples_alpha = None
        if 'beta1' in remove:
            self._joint_samples_beta1 = None
        if 'beta2' in remove:
            self._joint_samples_beta2 = None
        if 'sigma' in remove:
            self._joint_samples_sigma = None
        if 'X1' in remove:
            self._joint_samples_X1 = None
        if 'X2' in remove:
            self._joint_samples_X2 = None
        if 'Y' in remove:
            self._joint_samples_Y = None
        return ()

    def _conditionout(self, keep, remove):
        names = remove
        fields = self.fields if names is None else self.byname(names)
        cond_domains = [field['domain'] for field in fields]
        return (cond_domains)

    def _density(self, x):

        X = np.array([self._joint_samples_alpha,self._joint_samples_beta1,
                      self._joint_samples_beta2, self._joint_samples_sigma,
                      self._joint_samples_X1, self._joint_samples_X2,
                      self._joint_samples_mu]).T
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
        x = np.reshape(x,(1,len(x)))
        logdensity = kde.score_samples(x)
        return (np.exp(logdensity))

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



