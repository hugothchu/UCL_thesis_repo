from abc import ABC, abstractmethod
import numpy as np
import sklearn as sk
from copy import deepcopy
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import BallTree
from MICERegressor import MICELinearRegressor
from sklearn.preprocessing import Imputer
import fancyimpute as fi

class SupervisedImputation(ABC):
	@abstractmethod
	def train(self, df_gt):
		raise NotImplementedError('Abstract method "impute" not implemented in base class')

	@abstractmethod
	def impute(self, df_missing):
		raise NotImplementedError('Abstract method "impute" not implemented in base class')

class ImputeWithTrainedGlobalMean(SupervisedImputation):
	def train(self, df_gt):
		self.global_mean = np.nanmean(df_gt.as_matrix())

	def impute(self, df_missing):
		imputed_data = deepcopy(df_missing.as_matrix())
		missing_indices = np.isnan(imputed_data)
		imputed_data[missing_indices] = self.global_mean
		return imputed_data

class ImputeWithTrainedFeatureMean(SupervisedImputation):
	def train(self, df_gt):
		self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
		self.imputer.fit(df_gt.as_matrix())

	def impute(self, df_missing):
		return self.imputer.transform(df_missing.as_matrix())

class ImputeWithSupervisedKNN(SupervisedImputation):
	def train(self, df_gt):
		mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
		mean_imputer.fit(df_gt.as_matrix())
		self.imputer = BallTree(mean_imputer.transform(df_gt.as_matrix()))

	def impute(self, df_missing):
		missing_array = df_missing.as_matrix()
		missing_mask = df_missing.isnull().as_matrix()
		mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
		mean_imputer.fit(missing_array)
		missing_array = mean_imputer.transform(missing_array)
		dist, ind = self.imputer.query(missing_array, k=5)

		for i in range(ind.shape[0]):
			prediction = 0
			distance = 0
			for j in range(ind.shape[1]):
				prediction = prediction + self.imputer.data[ind[i, j]] * dist[i, j]
				distance = distance + dist[i, j]

			prediction = prediction / distance
			missing_array[i, missing_mask[i]] = prediction[missing_mask[i]]

		mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
		mean_imputer.fit(missing_array)
		return mean_imputer.transform(missing_array)

class ImputeWithSupervisedChainedEquations(SupervisedImputation):
	def __init__(self, regressor_type='Linear'):
		self.regressor_type = regressor_type

	def train(self, df_gt):
		df_gt = df_gt.as_matrix()
		df_gt = np.nan_to_num(df_gt)
		self.D = df_gt.shape[1]
		self.regressors = []

		for d in range(self.D):
			regressor = LinearRegression() if self.regressor_type == 'Linear' else RandomForestRegressor(n_estimators=600, max_features=0.3, min_impurity_decrease=0.1, n_jobs=-1)			
			regressor.fit(df_gt[:, np.arange(self.D) != d], df_gt[:, d])
			self.regressors.append(regressor)

	def impute(self, df_missing):
		imputed_data = df_missing.as_matrix()
		missing_indices = np.isnan(imputed_data)
		imputed_data = np.nan_to_num(imputed_data)

		for d in range(self.D):
			if np.any(missing_indices[:, d]):
				imputed_data[missing_indices[:, d], d] = self.regressors[d].predict(imputed_data[missing_indices[:, d]][:, np.arange(self.D) != d])

		return imputed_data

class UnsupervisedImputation(ABC):
	def __init__(self, missing_data):
		self.missing_data = missing_data.as_matrix()
		self.missing_indices = np.isnan(self.missing_data)
		self.non_missing_indices = ~self.missing_indices

	@abstractmethod
	def impute(self):
		raise NotImplementedError('Abstract method "impute" not implemented in base class')

class ImputeWithGlobalMean(UnsupervisedImputation):
	def impute(self):
		imputed_data = deepcopy(self.missing_data)
		imputed_data[self.missing_indices] = np.nanmean(self.missing_data)
		return imputed_data

class ImputeWithFeatureMean(UnsupervisedImputation):
	def impute(self):
		imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
		imputer.fit(self.missing_data)
		return imputer.transform(self.missing_data)

class ImputeWithUnsupervisedKNN(UnsupervisedImputation):
	def impute(self):
		return fi.KNN(verbose=False).complete(self.missing_data)

class ImputeWithMatrixSVD(UnsupervisedImputation):
	def impute(self):
		return fi.SoftImpute(verbose=False).complete(self.missing_data)

class ImputeWithMICE(UnsupervisedImputation):
	def __init__(self, missing_data, regressor=None, convergence_limit=0.01, verbose=False, window=None):
		super(ImputeWithMICE, self).__init__(missing_data)
		
		self.convergence_limit = convergence_limit
		self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
		self.imputer.fit(self.missing_data)

		if regressor is None:
			self.regressor = MICELinearRegressor()
		else:
			self.regressor = regressor

		self.old_array = None
		self.verbose = verbose
		self.delta_list = []

		self.N = missing_data.shape[0]
		self.D = missing_data.shape[1]

		if window == None:
			self.window = self.D
		else:
			self.window = window

	def _running_mean(self, x, N):
		return np.mean(x[-N:])

	def _initialize(self):
		self.k = 0
		self.old_array = self.imputer.transform(self.missing_data)
		return deepcopy(self.old_array)

	def _check_convergence(self, data_array, d):
		self.k = self.k + 1
		#delta = np.sqrt(np.sum((data_array[self.missing_indices] - self.old_array[self.missing_indices]) ** 2) / (self.N * self.D))
		#convergence_achieved = delta < self.convergence_limit

		delta = np.sqrt(np.sum((data_array[self.missing_indices[:, d], d] - self.old_array[self.missing_indices[:, d], d]) ** 2) / np.sum(self.missing_indices[:, d]))
		if np.isnan(delta):
			return False	# Column has no missingness
		self.delta_list.append(delta)
		window = len(self.delta_list) if len(self.delta_list) < self.window else self.window
		convergence_achieved = self._running_mean(np.array(self.delta_list), window) < self.convergence_limit

		if self.verbose:
			print('Iteration: {} Delta: {}'.format(self.k, delta))

		if ~convergence_achieved:
			self.old_array = deepcopy(data_array)
			return False
		else:
			return True

	def _fit_regression(self, endog, exog):
		self.regressor.fit(exog, endog)

	def _predict_with_regression(self, exog):
		return self.regressor.predict(exog)

	def impute(self):
		data_array = self._initialize()
		convergence = False

		d = 0
		while convergence is False:
			endog_all = data_array[:, d]
			exog_all = np.delete(data_array, d, axis=1)

			# Fit regression only on observed entries
			rows_for_regression = np.where(self.non_missing_indices[:, d])
			fit_endog = endog_all[rows_for_regression]
			fit_exog = np.squeeze(exog_all[rows_for_regression, :])
			self._fit_regression(fit_endog, fit_exog)

			# Perform imputation on missing entries
			rows_to_impute = self.missing_indices[:, d]
			impute_exog = exog_all[rows_to_impute, :]
			if impute_exog.shape[0] > 0:
				data_array[rows_to_impute, d] = self._predict_with_regression(impute_exog)

			convergence = self._check_convergence(data_array, d)
			d = (d + 1) % self.D

		print('MICE converged in {} iterations'.format(self.k))
		self.total_iterations = self.k
		self.k = 0
		return data_array

class AdaptiveMICE(ImputeWithMICE):
	def __init__(self, missing_data, regressor=None, convergence_limit=0.01, verbose=False, bound=5, step=0.05, decay=0.03, window=None):
		super(AdaptiveMICE, self).__init__(missing_data)
		
		missing_percentages = missing_data.isnull().sum() / missing_data.shape[1]
		self.random_scan_probabilities = missing_percentages / missing_percentages.sum()
		self.update_list = []
		self.step = step
		self.decay = decay
		self.verbose = verbose

		if window == None:
			self.window = self.D
		else:
			self.window = window

	def _check_convergence(self, data_array, d):
		self.k = self.k + 1

		delta = np.sqrt(np.sum((data_array[self.missing_indices[:, d], d] - self.old_array[self.missing_indices[:, d], d]) ** 2) / np.sum(self.missing_indices[:, d]))
		if np.isnan(delta):
			return False, False	# Column has no missingness

		self.delta_list.append(delta)
		window = len(self.delta_list) if len(self.delta_list) < self.window else self.window
		convergence_achieved = self._running_mean(np.array(self.delta_list), window) < self.convergence_limit

		if self.verbose:
			print('Iteration: {} Delta: {}'.format(self.k, delta))

		if ~convergence_achieved:
			self.old_array = deepcopy(data_array)
			return False, True
		else:
			return True, False

	def _update_scan_probabilities(self, last_updated_column):
		update_list = np.array(self.update_list)
		delta_list = np.array(self.delta_list)
		column_updates = delta_list[update_list == last_updated_column]

		if len(column_updates) >= 2:
			self._adjust_probabilities(last_updated_column, increase_d=column_updates[-1] > column_updates[-2])
			self.step = self.step * (1 - self.decay)

	def _adjust_probabilities(self, d, increase_d):
		column_d = self.random_scan_probabilities[d]
		remaining_columns = self.random_scan_probabilities[np.arange(self.D) != d]

		if increase_d:
			sorted_indices = np.argsort(remaining_columns)
			contributions = 0
			j = 0

			for i in sorted_indices:
				required_contribution = (self.step - contributions) / (self.D - j)

				if remaining_columns[i] - required_contribution >= 0:
					remaining_columns[i] = remaining_columns[i] - required_contribution
					contributions = contributions + required_contribution
				else:
					remaining_columns[i] = 0
					contributions = contributions + remaining_columns[i]

				j = j + 1

			column_d = column_d + contributions

		else:
			if column_d - self.step < 0:
				adjustment = column_d
				column_d = 0
			else:
				adjustment = self.step
				column_d = column_d - adjustment

			remaining_columns = remaining_columns + adjustment / (self.D - 1)

		self.random_scan_probabilities[d] = column_d
		self.random_scan_probabilities[np.arange(self.D) != d] = remaining_columns

		# adjust for numerical drift
		if abs(np.sum(self.random_scan_probabilities) - 1.0) > 1e-10:
			self.random_scan_probabilities = self.random_scan_probabilities / np.sum(self.random_scan_probabilities)

	def impute(self):
		data_array = self._initialize()
		N, D = data_array.shape
		convergence = False

		while convergence is False:
			# pick column to update
			d = np.squeeze(np.random.choice(np.arange(D), size=1, p=self.random_scan_probabilities))

			endog_all = data_array[:, d]
			exog_all = np.delete(data_array, d, axis=1)

			# Fit regression only on observed entries
			rows_for_regression = self.non_missing_indices[:, d]
			fit_endog = endog_all[rows_for_regression]
			fit_exog = exog_all[rows_for_regression, :]
			self._fit_regression(fit_endog, fit_exog)

			# Perform imputation on missing entries
			rows_to_impute = self.missing_indices[:, d]
			impute_exog = exog_all[rows_to_impute, :]
			if impute_exog.shape[0] > 0:
				data_array[rows_to_impute, d] = self._predict_with_regression(impute_exog)

			convergence, update_probs = self._check_convergence(data_array, d)
			
			if update_probs:
				self.update_list.append(d)
				self._update_scan_probabilities(d)

		print('Adaptive MICE converged in {} iterations'.format(len(self.delta_list)))
		self.total_iterations = len(self.delta_list)
		return data_array
