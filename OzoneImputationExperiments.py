import pandas as pd
import numpy as np
from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ImputationStrategyEvaluation import evaluate_supervised_imputations, evaluate_unsupervised_imputations, evaluate_pipeline_losses
from sklearn.linear_model import LogisticRegression

df_ozone = pd.read_csv('./eighthr.csv', header=None, prefix='C')
df_oday = df_ozone['C73']
df_ozone = df_ozone.drop('C73', 1)
df_ozone = df_ozone.drop('C0', 1)
df_ozone = df_ozone.replace('?', np.NaN)
df_ozone = df_ozone.astype(np.float32, errors='ignore')

# Normalize all experiments
df_ozone = (df_ozone - df_ozone.mean()) / (df_ozone.max() - df_ozone.min())

N, D = df_ozone.shape

original_mask = df_ozone.isnull()

def converse_nonimplication(original_mask, new_mask):
	return ~original_mask * new_mask

def create_MCAR_dataset(df, missing_probability=0.33):
	# Entries are missing at random
	mask = np.random.binomial(1, missing_probability, (N, D))
	df = deepcopy(df)
	df = df * (1 - mask)
	df[df == 0] = np.nan
	return df, converse_nonimplication(original_mask.as_matrix(), mask)

def create_MAR_dataset(df, missing_probability=0.4):
	# Length, Diameter and Height have higher missing probability if Height > 0.14
	df = deepcopy(df)
	condition = df['C70'] > df['C70'].mean()
	M = len(df[condition])
	missing_columns = ['C3', 'C5', 'C6', 'C8', 'C11', 'C22', 'C34', 'C45', 'C46', 'C57', 'C58', 'C69']
	mask = np.random.binomial(1, missing_probability, (M, len(missing_columns)))
	df.loc[condition, missing_columns] = df[condition][missing_columns] * (1 - mask)
	df[df == 0] = np.nan

	final_mask = pd.DataFrame(np.zeros(df.shape), columns=df.columns)
	final_mask.loc[condition, missing_columns] = converse_nonimplication(original_mask[condition][missing_columns].as_matrix(), mask)
	return df, final_mask

def create_MNAR_dataset(df, missing_probability=0.33):
	# Entries may be missing if certain it exceeds certain value
	df = deepcopy(df)
	missing_columns = ['C3', 'C5', 'C6', 'C8', 'C11', 'C22', 'C34', 'C45', 'C46', 'C57', 'C58', 'C69']
	condition = df[missing_columns] > df[missing_columns].mean()
	not_condition = ~condition
	mask = np.random.binomial(1, missing_probability, condition.shape)
	apply_mask = mask * condition + not_condition
	df[missing_columns] = df[missing_columns] * apply_mask
	df[df == 0] = np.nan

	final_mask = pd.DataFrame(np.zeros(df.shape), columns=df.columns)
	final_mask[missing_columns] = converse_nonimplication(original_mask[missing_columns].as_matrix(), mask)
	return df, converse_nonimplication(original_mask[missing_columns].as_matrix(), mask)

def create_Replicated_Missingness(df):
	df_pred = df.dropna(axis=0, how='any')
	df_reg = df.apply(lambda x: x.fillna(x.mean()),axis=0)

	mask_probabilities = []

	for d in range(D):
		regressor = LogisticRegression()
		regressor.fit(df_reg.loc[:, np.arange(D) != d], original_mask[original_mask.columns[d]])
		probabilities = regressor.predict_proba(df_pred.loc[:, np.arange(D) != d])
		mask_probabilities.append(probabilities[:, 1])

	mask_probabilities = np.array(mask_probabilities).T
	mask = np.random.binomial(1, mask_probabilities)
	df_pred = df_pred * (1 - mask)
	df_pred[df_pred == 0] = np.nan
	return df_pred, df_pred.isnull()

df_mcar, mcar_mask = create_MCAR_dataset(df_ozone)
df_mar, mar_mask = create_MAR_dataset(df_ozone)
df_mnar, mnar_mask = create_MNAR_dataset(df_ozone)
df_rm, rm_mask = create_Replicated_Missingness(df_ozone)

def evaluate_imputation_RMSE():

	print('.................. Evaluating MCAR case ..................')
	evaluate_supervised_imputations(df_ozone, df_mcar, mcar_mask)
	evaluate_unsupervised_imputations(df_ozone, df_mcar, mcar_mask)
	
	print('.................. Evaluating MAR case ..................')
	evaluate_supervised_imputations(df_ozone, df_mar, mar_mask)
	evaluate_unsupervised_imputations(df_ozone, df_mar, mar_mask)

	print('.................. Evaluating MNAR case ..................')
	evaluate_supervised_imputations(df_ozone, df_mnar, mnar_mask)
	evaluate_unsupervised_imputations(df_ozone, df_mnar, mnar_mask)

	print('.................. Evaluating Replicated Missingness case ..................')
	evaluate_supervised_imputations(df_ozone.dropna(axis=0, how='any'), df_rm, rm_mask)
	evaluate_unsupervised_imputations(df_ozone.dropna(axis=0, how='any'), df_rm, rm_mask)

def evaluate_imputation_RMSE():
	print('Evaluating MCAR case ...')
	array_rmse_1 = []
	array_rmse_2 = []
	array_rmse_3 = []
	array_rmse_4 = []
	for _ in range(100):
		rmse_1, rmse_2, rmse_3, rmse_4 = evaluate_supervised_imputations(df_ozone, df_mcar, mcar_mask)
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
		rmse_1, rmse_2, rmse_3, rmse_4, rmse_5, rmse_6, iters_1, iters_2 = evaluate_unsupervised_imputations(df_ozone, df_mcar, mcar_mask)
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
		rmse_1, rmse_2, rmse_3, rmse_4 = evaluate_supervised_imputations(df_ozone, df_mar, mar_mask)
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
		rmse_1, rmse_2, rmse_3, rmse_4, rmse_5, rmse_6, iters_1, iters_2 = evaluate_unsupervised_imputations(df_ozone, df_mar, mar_mask)
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
		rmse_1, rmse_2, rmse_3, rmse_4 = evaluate_supervised_imputations(df_ozone, df_mnar, mnar_mask)
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
		rmse_1, rmse_2, rmse_3, rmse_4, rmse_5, rmse_6, iters_1, iters_2 = evaluate_unsupervised_imputations(df_ozone, df_mnar, mnar_mask)
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

	print('Evaluating Complete case ...')
	array_rmse_1 = []
	array_rmse_2 = []
	array_rmse_3 = []
	array_rmse_4 = []
	for _ in range(100):
		rmse_1, rmse_2, rmse_3, rmse_4 = evaluate_supervised_imputations(df_ozone.dropna(axis=0, how='any'), df_rm, rm_mask)
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
		rmse_1, rmse_2, rmse_3, rmse_4, rmse_5, rmse_6, iters_1, iters_2 = evaluate_unsupervised_imputations(df_ozone.dropna(axis=0, how='any'), df_rm, rm_mask)
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


le = LabelEncoder()
print('.................. Evaluating MCAR case ..................')
y = df_oday.iloc[df_mcar.index]
le.fit(df_oday.unique())
y = le.transform(y)
evaluate_pipeline_losses(df_mcar, y, 'MCAR')
print('.................. Evaluating MAR case ..................')
y = df_oday.iloc[df_mar.index]
le.fit(df_oday.unique())
y = le.transform(y)
evaluate_pipeline_losses(df_mar, y, 'MAR')
print('.................. Evaluating MNAR case ..................')
y = df_oday.iloc[df_mnar.index]
le.fit(df_oday.unique())
y = le.transform(y)
evaluate_pipeline_losses(df_mnar, y, 'MNAR')
print('.................. Evaluating Complete case ..................')
y = df_oday.iloc[df_ozone.index]
le.fit(df_oday.unique())
y = le.transform(y)
evaluate_pipeline_losses(df_ozone, y, 'RM')
