# Takes an incomplete feature matrix and a complete label vector, impute the feature matrix, then split into train-test sets
# Perform supervised learning procedures on the fully imputed feature matrix to predict the labels

class ImputePredictAnalyzePipeline():
	def __init__(self, imputation_strategy, prediction_strategy, train_test_splitter, endog):
		self.imputation_strategy = imputation_strategy
		self.prediction_strategy = prediction_strategy
		self.train_test_splitter = train_test_splitter
		self.endog = endog

	def impute_and_fit(self, training_set=None):
		if training_set is None:
			imputed_exog = self.imputation_strategy.impute()
		else:
			self.imputation_strategy.train(training_set)
			imputed_exog = self.imputation_strategy.impute(training_set)

		exog_train, exog_test, endog_train, endog_test = self.train_test_splitter.split(imputed_exog, self.endog, test_size=0.33)
		self.prediction_strategy.fit(exog_train, endog_train)
		pred_endog_proba, pred_endog = self.prediction_strategy.predict(exog_test)
		return self.prediction_strategy.evaluate_loss(endog_test, pred_endog_proba, pred_endog)
