from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class TrainTestSplitter(ABC):
	@abstractmethod
	def split(self, exog, endog, test_size):
		raise NotImplementedError('Method "split" not implemented in base class')

class AbaloneDataSetTrainTestSplitter(TrainTestSplitter):
	def split(self, exog, endog, test_size):
		exog_train, exog_test, endog_train, endog_test = train_test_split(exog, endog, test_size=test_size)
		return(exog_train, exog_test, endog_train, endog_test)

class EGymDataSetTraintestSplitter(TrainTestSplitter):
	def split(self, exog, endog, test_size):
		exog_train, exog_test, endog_train, endog_test = train_test_split(exog, endog, test_size=test_size)
		return(exog_train, exog_test, endog_train, endog_test)
		