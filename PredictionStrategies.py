from abc import ABC, abstractmethod
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import log_loss, accuracy_score
import numpy as np

# Wrapper class for various predictive experiments
class PredictionStrategy(ABC):
	@abstractmethod
	def fit(self, exog, endog):
		raise NotImplementedError('Abstract method "fit" not implemented in base class')

	@abstractmethod
	def predict(self, exog):
		raise NotImplementedError('Abstract method "predict" not implemented in base class')

	@abstractmethod
	def evaluate_loss(self, gt_endog, pred_endog):
		raise NotImplementedError('Abstract method "predict" not implemented in base class')

class DummyRegressionPrediction(PredictionStrategy):
	def __init__(self):
		self.predictor = DummyClassifier(strategy='most_frequent')

	def fit(self, exog, endog):
		self.predictor.fit(exog, endog)

	def predict(self, exog):
		return self.predictor.predict_proba(exog), self.predictor.predict(exog)

	def evaluate_loss(self, gt_endog, pred_endog_proba, pred_endog):
		return log_loss(gt_endog, pred_endog_proba), accuracy_score(gt_endog, pred_endog)

class LogisticRegressionPrediction(PredictionStrategy):
	def __init__(self):
		self.predictor = LogisticRegression()

	def fit(self, exog, endog):
		self.predictor.fit(exog, endog)

	def predict(self, exog):
		return self.predictor.predict_proba(exog), self.predictor.predict(exog)

	def evaluate_loss(self, gt_endog, pred_endog_proba, pred_endog):
		return log_loss(gt_endog, pred_endog_proba), accuracy_score(gt_endog, pred_endog)
