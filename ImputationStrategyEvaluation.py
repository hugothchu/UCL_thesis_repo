import numpy as np
from sklearn.model_selection import train_test_split
from MICERegressor import MICERandomForestRegressor, MICEGlobalLocalRegressor
from ImputationStrategies import ImputeWithTrainedGlobalMean, ImputeWithTrainedFeatureMean, ImputeWithSupervisedKNN, ImputeWithSupervisedChainedEquations, ImputeWithGlobalMean, ImputeWithFeatureMean, ImputeWithUnsupervisedKNN, ImputeWithMatrixSVD, ImputeWithMICE, AdaptiveMICE
from PredictionStrategies import DummyRegressionPrediction, LogisticRegressionPrediction
from TrainTestSplitters import AbaloneDataSetTrainTestSplitter
from ImputePredictAnalyzePipeline import ImputePredictAnalyzePipeline

def get_rmse(ground_truth, df_masked, imputed_matrix, original_mask=None):
	if original_mask is None:
		mask = df_masked.isnull()
	else:
		mask = original_mask.astype('bool')

	return np.sqrt(np.nanmean(((ground_truth - imputed_matrix)[mask] ** 2).as_matrix()))

def evaluate_supervised_imputations(df_gt, df_missing, original_mask=None):
	if original_mask is None:
		_, missing_test, gt_train, gt_test = train_test_split(df_missing, df_gt, test_size=0.33, random_state=1)
		original_mask_test = None
	else:
		_, missing_test, gt_train, gt_test, _, original_mask_test = train_test_split(df_missing, df_gt, original_mask, test_size=0.33)

	imputer_1 = ImputeWithTrainedGlobalMean()
	imputer_1.train(gt_train)

	imputer_2 = ImputeWithTrainedFeatureMean()
	imputer_2.train(gt_train)

	imputer_3 = ImputeWithSupervisedChainedEquations()
	imputer_3.train(gt_train)

	imputer_4 = ImputeWithSupervisedChainedEquations(regressor_type='RF')
	imputer_4.train(gt_train)

	#imputer_5 = ImputeWithSupervisedKNN()
	#imputer_5.train(gt_train)
	
	full_1 = imputer_1.impute(missing_test)
	full_2 = imputer_2.impute(missing_test)
	full_3 = imputer_3.impute(missing_test)
	full_4 = imputer_4.impute(missing_test)
	#full_5 = imputer_5.impute(missing_test)

	rmse_1 = get_rmse(gt_test, missing_test, full_1, original_mask_test)
	rmse_2 = get_rmse(gt_test, missing_test, full_2, original_mask_test)
	rmse_3 = get_rmse(gt_test, missing_test, full_3, original_mask_test)
	rmse_4 = get_rmse(gt_test, missing_test, full_4, original_mask_test)

	'''
	print('RMSE for Global Mean Imputation: {}'.format(rmse_1))
	print('RMSE for Feature Mean Imputation: {}'.format(rmse_2))
	print('RMSE for Chained Equations Imputation: {}'.format(rmse_3))
	print('RMSE for Chained RFs Imputation: {}'.format(rmse_4))
	#print('RMSE for Supervised KNN Imputation: {}'.format(get_rmse(gt_test, missing_test, full_5, original_mask_test)))
	'''

	return rmse_1, rmse_2, rmse_3, rmse_4

def evaluate_unsupervised_imputations(df_gt, df_missing, original_mask=None):
	if original_mask is None:
		df_missing, _, df_gt, _ = train_test_split(df_missing, df_gt, test_size=0.33)
	else:
		df_missing, _, df_gt, _, original_mask, _ = train_test_split(df_missing, df_gt, original_mask, test_size=0.33)

	imputer_1 = ImputeWithGlobalMean(df_missing)
	imputer_2 = ImputeWithFeatureMean(df_missing)
	imputer_3 = ImputeWithUnsupervisedKNN(df_missing)
	imputer_4 = ImputeWithMatrixSVD(df_missing)
	imputer_5 = ImputeWithMICE(df_missing, convergence_limit=0.01)
	#imputer_6 = ImputeWithMICE(df_missing, regressor=MICERandomForestRegressor(), convergence_limit=0.001)
	imputer_6 = AdaptiveMICE(df_missing, convergence_limit=0.01)
	#imputer_8 = AdaptiveMICE(df_missing, regressor=MICEGlobalLocalRegressor(), convergence_limit=0.0001)
	
	full_1 = imputer_1.impute()
	full_2 = imputer_2.impute()
	full_3 = imputer_3.impute()
	full_4 = imputer_4.impute()
	full_5 = imputer_5.impute()
	#full_6 = imputer_6.impute()
	full_6 = imputer_6.impute()
	#full_8 = imputer_8.impute()

	rmse_1 = get_rmse(df_gt, df_missing, full_1, original_mask)
	rmse_2 = get_rmse(df_gt, df_missing, full_2, original_mask)
	rmse_3 = get_rmse(df_gt, df_missing, full_3, original_mask)
	rmse_4 = get_rmse(df_gt, df_missing, full_4, original_mask)
	rmse_5 = get_rmse(df_gt, df_missing, full_5, original_mask)
	rmse_6 = get_rmse(df_gt, df_missing, full_6, original_mask)

	iterations_1 = imputer_5.total_iterations
	iterations_2 = imputer_6.total_iterations

	'''
	print('RMSE for Global Mean Imputation: {}'.format(rmse_1))
	print('RMSE for Feature Mean Imputation: {}'.format(get_rmse(df_gt, df_missing, full_2, original_mask)))
	print('RMSE for KNN Imputation: {}'.format(get_rmse(df_gt, df_missing, full_3, original_mask)))
	print('RMSE for Matrix SVD Imputation: {}'.format(get_rmse(df_gt, df_missing, full_4, original_mask)))
	print('RMSE for MICE LR Imputation: {}'.format(get_rmse(df_gt, df_missing, full_5, original_mask)))
	#print('RMSE for MICE RF Imputation: {}'.format(get_rmse(df_gt, df_missing, full_6, original_mask)))
	print('RMSE for Adaptive MICE Imputation: {}'.format(get_rmse(df_gt, df_missing, full_7, original_mask)))
	#print('RMSE for Adaptive MICE Global-Local Imputation: {}'.format(get_rmse(df_gt, df_missing, full_8, original_mask)))
	'''

	return rmse_1, rmse_2, rmse_3, rmse_4, rmse_5, rmse_6, iterations_1, iterations_2

def evaluate_pipeline_losses(df_missing, y, name):
	pipeline = ImputePredictAnalyzePipeline(ImputeWithTrainedGlobalMean(), LogisticRegressionPrediction(), AbaloneDataSetTrainTestSplitter(), y)
	loss_1a, loss_1b = pipeline.impute_and_fit(df_missing)

	pipeline = ImputePredictAnalyzePipeline(ImputeWithTrainedFeatureMean(), LogisticRegressionPrediction(), AbaloneDataSetTrainTestSplitter(), y)
	loss_2a, loss_2b = pipeline.impute_and_fit(df_missing)

	pipeline = ImputePredictAnalyzePipeline(ImputeWithSupervisedChainedEquations(), LogisticRegressionPrediction(), AbaloneDataSetTrainTestSplitter(), y)
	loss_3a, loss_3b = pipeline.impute_and_fit(df_missing)

	pipeline = ImputePredictAnalyzePipeline(ImputeWithSupervisedChainedEquations(regressor_type='RF'), LogisticRegressionPrediction(), AbaloneDataSetTrainTestSplitter(), y)
	loss_4a, loss_4b = pipeline.impute_and_fit(df_missing)

	pipeline = ImputePredictAnalyzePipeline(ImputeWithGlobalMean(df_missing), LogisticRegressionPrediction(), AbaloneDataSetTrainTestSplitter(), y)
	loss_5a, loss_5b = pipeline.impute_and_fit()

	pipeline = ImputePredictAnalyzePipeline(ImputeWithFeatureMean(df_missing), LogisticRegressionPrediction(), AbaloneDataSetTrainTestSplitter(), y)
	loss_6a, loss_6b = pipeline.impute_and_fit()

	pipeline = ImputePredictAnalyzePipeline(ImputeWithUnsupervisedKNN(df_missing), LogisticRegressionPrediction(), AbaloneDataSetTrainTestSplitter(), y)
	loss_7a, loss_7b = pipeline.impute_and_fit()

	pipeline = ImputePredictAnalyzePipeline(ImputeWithMatrixSVD(df_missing), LogisticRegressionPrediction(), AbaloneDataSetTrainTestSplitter(), y)
	loss_8a, loss_8b = pipeline.impute_and_fit()

	pipeline = ImputePredictAnalyzePipeline(ImputeWithMICE(df_missing), LogisticRegressionPrediction(), AbaloneDataSetTrainTestSplitter(), y)
	loss_9a, loss_9b = pipeline.impute_and_fit()

	pipeline = ImputePredictAnalyzePipeline(AdaptiveMICE(df_missing, LogisticRegressionPrediction()), LogisticRegressionPrediction(), AbaloneDataSetTrainTestSplitter(), y)
	loss_10a, loss_10b = pipeline.impute_and_fit()

	print('Global Mean & Supervised & {} & {}'.format(loss_1a, loss_1b))
	print('Feature Mean & Supervised & {} & {}'.format(loss_2a, loss_2b))
	print('Linear Regression & Supervised & {} & {}'.format(loss_3a, loss_3b))
	print('Random Forests & Supervised & {} & {}'.format(loss_4a, loss_4b))
	print('K-NN & Supervised & {} & {}'.format(loss_7a, loss_7b))
	print('Global Mean & Unsupervised & {} & {}'.format(loss_5a, loss_5b))
	print('Feature Mean & Unsupervised & {} & {}'.format(loss_6a, loss_6b))
	print('Matrix SVD & Unsupervised & {} & {}'.format(loss_8a, loss_8b))
	print('MICE & Unsupervised & {} & {}'.format(loss_9a, loss_9b))
	print('Adaptive MICE & Unsupervised & {} & {}'.format(loss_10a, loss_10b))
