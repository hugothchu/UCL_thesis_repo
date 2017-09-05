import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from ImputationStrategyEvaluation import evaluate_supervised_imputations, evaluate_unsupervised_imputations, evaluate_pipeline_losses
from ImputationStrategies import ImputeWithTrainedGlobalMean, ImputeWithTrainedFeatureMean, ImputeWithSupervisedKNN, ImputeWithSupervisedChainedEquations, ImputeWithGlobalMean, ImputeWithFeatureMean, ImputeWithUnsupervisedKNN, ImputeWithMatrixSVD, ImputeWithMICE, AdaptiveMICE
from PredictionStrategies import DummyRegressionPrediction, LogisticRegressionPrediction
from TrainTestSplitters import AbaloneDataSetTrainTestSplitter
from ImputePredictAnalyzePipeline import ImputePredictAnalyzePipeline
from MICERegressor import MICERandomForestRegressor

df_abalone = pd.read_csv('./abalone.csv', names=['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings'])

y = df_abalone['Sex']
le = LabelEncoder()
le.fit(y.unique())
y = le.transform(y)

df_abalone = df_abalone.drop('Sex', 1)

df_abalone = (df_abalone - df_abalone.mean()) / (df_abalone.max() - df_abalone.min())

N, D = df_abalone.shape

def create_MCAR_dataset(df, missing_probability=0.33):
	# Entries are missing at random
	mask = np.random.binomial(1, missing_probability, (N, D))
	df = deepcopy(df)
	df = df * (1 - mask)
	df[df == 0] = np.nan
	return df

def create_MAR_dataset(df, missing_probability=0.66):
	# Length, Diameter and Height have higher missing probability if Height > 0.14
	df = deepcopy(df)
	condition = df['Height'] > df['Height'].mean()
	M = len(df[condition])
	mask = np.random.binomial(1, missing_probability, (M, 5))
	df.loc[condition, ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight']] = df[condition][['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight']] * (1 - mask)
	df[df == 0] = np.nan
	return df

def create_MNAR_dataset(df, missing_probability=0.33):
	# Entries may be missing if certain it exceeds certain value
	df = deepcopy(df)
	condition = df[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight']] > df[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight']].mean()
	not_condition = ~condition
	mask = np.random.binomial(1, missing_probability, condition.shape)
	mask = mask * condition + not_condition
	df[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight']] = df[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight']] * mask
	df[df == 0] = np.nan
	return df

df_mcar = create_MCAR_dataset(df_abalone)
df_mar = create_MAR_dataset(df_abalone)
df_mnar = create_MNAR_dataset(df_abalone)

def evaluate_imputation_RMSE():
	print('Evaluating MCAR case ...')
	array_rmse_1 = []
	array_rmse_2 = []
	array_rmse_3 = []
	array_rmse_4 = []
	for _ in range(100):
		rmse_1, rmse_2, rmse_3, rmse_4 = evaluate_supervised_imputations(df_abalone, df_mcar)
		array_rmse_1.append(rmse_1)
		array_rmse_2.append(rmse_2)
		array_rmse_3.append(rmse_3)
		array_rmse_4.append(rmse_4)
	print('Global Mean & Supervised & {} & {}'.format(np.mean(array_rmse_1), np.std(array_rmse_1)))
	print('Feature Mean & Supervised & {} & {}'.format(np.mean(array_rmse_2), np.std(array_rmse_2)))
	print('Linear Regression & Supervised & {} & {}'.format(np.mean(array_rmse_3), np.std(array_rmse_3)))
	print('Random Forests & Supervised & {} & {}'.format(np.mean(array_rmse_4), np.std(array_rmse_4)))
	array_rmse_1 = []
	array_rmse_2 = []
	array_rmse_3 = []
	array_rmse_4 = []
	array_rmse_5 = []
	array_rmse_6 = []
	array_iters_1 = []
	array_iters_2 = []
	for _ in range(100):
		rmse_1, rmse_2, rmse_3, rmse_4, rmse_5, rmse_6, iters_1, iters_2 = evaluate_unsupervised_imputations(df_abalone, df_mcar)
		array_rmse_1.append(rmse_1)
		array_rmse_2.append(rmse_2)
		array_rmse_3.append(rmse_3)
		array_rmse_4.append(rmse_4)
		array_rmse_5.append(rmse_5)
		array_rmse_6.append(rmse_6)
		array_iters_1.append(iters_1)
		array_iters_2.append(iters_2)
	print('K-NN & Supervised & {} & {}'.format(np.mean(array_rmse_3), np.std(array_rmse_3)))
	print('Global Mean & Unsupervised & {} & {}'.format(np.mean(array_rmse_1), np.std(array_rmse_1)))
	print('Feature Mean & Unsupervised & {} & {}'.format(np.mean(array_rmse_2), np.std(array_rmse_2)))
	print('Matrix SVD & Unsupervised & {} & {}'.format(np.mean(array_rmse_4), np.std(array_rmse_4)))
	print('MICE & Unsupervised & {} & {}'.format(np.mean(array_rmse_5), np.std(array_rmse_5)))
	print('Adaptive MICE & Unsupervised & {} & {}'.format(np.mean(array_rmse_6), np.std(array_rmse_6)))
	print('MICE Iterations & {} & {}'.format(np.mean(array_iters_1), np.std(array_iters_1)))
	print('Adaptive MICE Iterations & {} & {}'.format(np.mean(array_iters_2), np.std(array_iters_2)))

	print('Evaluating MAR case ...')
	array_rmse_1 = []
	array_rmse_2 = []
	array_rmse_3 = []
	array_rmse_4 = []
	for _ in range(100):
		rmse_1, rmse_2, rmse_3, rmse_4 = evaluate_supervised_imputations(df_abalone, df_mar)
		array_rmse_1.append(rmse_1)
		array_rmse_2.append(rmse_2)
		array_rmse_3.append(rmse_3)
		array_rmse_4.append(rmse_4)
	print('Global Mean & Supervised & {} & {}'.format(np.mean(array_rmse_1), np.std(array_rmse_1)))
	print('Feature Mean & Supervised & {} & {}'.format(np.mean(array_rmse_2), np.std(array_rmse_2)))
	print('Linear Regression & Supervised & {} & {}'.format(np.mean(array_rmse_3), np.std(array_rmse_3)))
	print('Random Forests & Supervised & {} & {}'.format(np.mean(array_rmse_4), np.std(array_rmse_4)))
	array_rmse_1 = []
	array_rmse_2 = []
	array_rmse_3 = []
	array_rmse_4 = []
	array_rmse_5 = []
	array_rmse_6 = []
	array_iters_1 = []
	array_iters_2 = []
	for _ in range(100):
		rmse_1, rmse_2, rmse_3, rmse_4, rmse_5, rmse_6, iters_1, iters_2 = evaluate_unsupervised_imputations(df_abalone, df_mar)
		array_rmse_1.append(rmse_1)
		array_rmse_2.append(rmse_2)
		array_rmse_3.append(rmse_3)
		array_rmse_4.append(rmse_4)
		array_rmse_5.append(rmse_5)
		array_rmse_6.append(rmse_6)
		array_iters_1.append(iters_1)
		array_iters_2.append(iters_2)
	print('K-NN & Supervised & {} & {}'.format(np.mean(array_rmse_3), np.std(array_rmse_3)))
	print('Global Mean & Unsupervised & {} & {}'.format(np.mean(array_rmse_1), np.std(array_rmse_1)))
	print('Feature Mean & Unsupervised & {} & {}'.format(np.mean(array_rmse_2), np.std(array_rmse_2)))
	print('Matrix SVD & Unsupervised & {} & {}'.format(np.mean(array_rmse_4), np.std(array_rmse_4)))
	print('MICE & Unsupervised & {} & {}'.format(np.mean(array_rmse_5), np.std(array_rmse_5)))
	print('Adaptive MICE & Unsupervised & {} & {}'.format(np.mean(array_rmse_6), np.std(array_rmse_6)))
	print('MICE Iterations & {} & {}'.format(np.mean(array_iters_1), np.std(array_iters_1)))
	print('Adaptive MICE Iterations & {} & {}'.format(np.mean(array_iters_2), np.std(array_iters_2)))

	print('Evaluating MNAR case ...')
	array_rmse_1 = []
	array_rmse_2 = []
	array_rmse_3 = []
	array_rmse_4 = []
	for _ in range(100):
		rmse_1, rmse_2, rmse_3, rmse_4 = evaluate_supervised_imputations(df_abalone, df_mnar)
		array_rmse_1.append(rmse_1)
		array_rmse_2.append(rmse_2)
		array_rmse_3.append(rmse_3)
		array_rmse_4.append(rmse_4)
	print('Global Mean & Supervised & {} & {}'.format(np.mean(array_rmse_1), np.std(array_rmse_1)))
	print('Feature Mean & Supervised & {} & {}'.format(np.mean(array_rmse_2), np.std(array_rmse_2)))
	print('Linear Regression & Supervised & {} & {}'.format(np.mean(array_rmse_3), np.std(array_rmse_3)))
	print('Random Forests & Supervised & {} & {}'.format(np.mean(array_rmse_4), np.std(array_rmse_4)))
	array_rmse_1 = []
	array_rmse_2 = []
	array_rmse_3 = []
	array_rmse_4 = []
	array_rmse_5 = []
	array_rmse_6 = []
	array_iters_1 = []
	array_iters_2 = []
	for _ in range(100):
		rmse_1, rmse_2, rmse_3, rmse_4, rmse_5, rmse_6, iters_1, iters_2 = evaluate_unsupervised_imputations(df_abalone, df_mnar)
		array_rmse_1.append(rmse_1)
		array_rmse_2.append(rmse_2)
		array_rmse_3.append(rmse_3)
		array_rmse_4.append(rmse_4)
		array_rmse_5.append(rmse_5)
		array_rmse_6.append(rmse_6)
		array_iters_1.append(iters_1)
		array_iters_2.append(iters_2)
	print('K-NN & Supervised & {} & {}'.format(np.mean(array_rmse_3), np.std(array_rmse_3)))
	print('Global Mean & Unsupervised & {} & {}'.format(np.mean(array_rmse_1), np.std(array_rmse_1)))
	print('Feature Mean & Unsupervised & {} & {}'.format(np.mean(array_rmse_2), np.std(array_rmse_2)))
	print('Matrix SVD & Unsupervised & {} & {}'.format(np.mean(array_rmse_4), np.std(array_rmse_4)))
	print('MICE & Unsupervised & {} & {}'.format(np.mean(array_rmse_5), np.std(array_rmse_5)))
	print('Adaptive MICE & Unsupervised & {} & {}'.format(np.mean(array_rmse_6), np.std(array_rmse_6)))
	print('MICE Iterations & {} & {}'.format(np.mean(array_iters_1), np.std(array_iters_1)))
	print('Adaptive MICE Iterations & {} & {}'.format(np.mean(array_iters_2), np.std(array_iters_2)))


evaluate_imputation_RMSE()
print('.................. Evaluating MCAR case ..................')
evaluate_pipeline_losses(df_mcar, y, 'MCAR')
print('.................. Evaluating MAR case ..................')
evaluate_pipeline_losses(df_mar, y, 'MAR')
print('.................. Evaluating MNAR case ..................')
evaluate_pipeline_losses(df_mnar, y, 'MNAR')
