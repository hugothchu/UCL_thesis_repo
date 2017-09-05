from abc import ABC, abstractmethod
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor

# Wrapper class for generic regressor for MICE Algorithm
class MICERegressor(ABC):
	@abstractmethod
	def fit(self, exog, endog):
		raise NotImplementedError('Abstract method "fit" not implemented in base class')

	@abstractmethod
	def predict(exog):
		raise NotImplementedError('Abstract method "predict" not implemented in base class')

class MICELinearRegressor(MICERegressor):
	def __init__(self):
		self.regressor = LinearRegression()

	def fit(self, exog, endog):
		self.regressor.fit(exog, endog)

	def predict(self, exog):
		return self.regressor.predict(exog)

class MICERandomForestRegressor(MICERegressor):
	def __init__(self, n_estimators=500, max_features=0.66):
		self.regressor = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, min_impurity_decrease=0.2, n_jobs=-1)

	def fit(self, exog, endog):
		self.regressor.fit(exog, endog)

	def predict(self, exog):
		return self.regressor.predict(exog)

class MICEGlobalLocalRegressor(MICERegressor):
	def __init__(self, weights=0.5, n_estimators=500, max_features=0.66, n_neighbors=400):
		self.global_regressor = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, min_impurity_decrease=0.2, n_jobs=-1)
		self.local_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
		self.weights = weights

	def fit(self, exog, endog):
		self.global_regressor.fit(exog, endog)
		self.local_regressor.fit(exog, endog)

	def predict(self, exog):
		return self.weights * self.global_regressor.predict(exog) + (1 - self.weights) * self.local_regressor.predict(exog)

'''
Dummy Regression - most_frequent t0 belief prob

Random Forest: Randomly impute columns

Imputation Error: Tune the imputatoin algorithm before feeding into the pipeline - resampled missingness - following the missing pattern

overlay missingness on the complete cases
uniformly remove points

pruning / tuning

remova GPR and adaboost

explain SL imputation - evaluate imputation similar to SL
'''