import numpy as np
import pandas as pd
import os, pickle
from MultiStrengthExploration import machine_measures

train_test_store = './train_test_store_'

def imputation_rmse(test_set, predictions):
	test_array = test_set[machine_measures].as_matrix()
	non_missing_indicators = ~np.isnan(test_array)
	total_non_missing = non_missing_indicators.sum()
	mse = (test_array - predictions) ** 2
	return np.sqrt(mse[non_missing_indicators].sum() / total_non_missing)

# Custom train-test split to ensure test set will not carry user ID not already in the training set
# Can't absolutely guarantee the length of training and test sets
# This function guarantees that the train / test split between multivariate and univariate analysis are exactly the same,
# hence giving a fair comparison of RMSE's
def train_test_split(df_multi_strength, test_size, random_state=1):
	store = train_test_store + str(random_state)
	
	if os.path.isfile(store):
		with open(store, 'rb') as handle:
			train_test = pickle.load(handle)
	else:
		df_multi_strength.index.rename('index', inplace=True)
		def sampling_function(group):
			group_len = len(group)
			if group_len >= 3:
				return group.iloc[np.random.choice(range(0, group_len), np.ceil(test_size * group_len).astype(int), replace=False)]

		multi_test_set = df_multi_strength.groupby('User_ID').apply(sampling_function)
		multi_training_set = df_multi_strength.drop(multi_test_set.index.to_frame()['index'])
		
		uni_test_set = multi_to_uni(multi_test_set)
		uni_training_set = multi_to_uni(multi_training_set)

		train_test = (multi_training_set, multi_test_set, uni_training_set, uni_test_set)

		with open(store, "wb") as handle:
			pickle.dump(train_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return train_test

def multi_to_uni(df_multi_strength):
	df_uni_strength = pd.melt(df_multi_strength, id_vars=['User_ID', 'Months_Since_First_Workout'], value_vars=machine_measures, value_name='value').dropna()
	df_uni_strength.rename(columns={'variable': 'Machine'}, inplace=True)
	return df_uni_strength
