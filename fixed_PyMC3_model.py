# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de), Jonas Gütter (jonas.aaron.guetter@uni-jena.de)

from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core import data_operations as data_op
from mb_modelbase.models_core import data_aggregation as data_aggr
import pymc3 as pm
import numpy as np
import pandas as pd
from mb_modelbase.models_core.empirical_model import EmpiricalModel

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
        # model fitting by using maximum a posteriori
        map_estimate = pm.find_MAP(model=basic_model)
        self._model_params = map_estimate
        return ()

    def _marginalizeout(self, keep, remove):
        # maybe something like this?
        #trace[keep]
        return ()

    def _conditionout(self, keep, remove):
        return ()

    def density(self, x):
        return ()

    def _sample(self):
        return ()

    def copy(self, name=None):
        return ()


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
probabilistic_model_instance._fit()

print(probabilistic_model_instance.fields)
###