from ImportData import import_data
from MultiStrengthExploration import machine_measures, get_all_multi_strengths, get_multi_strengths
from Helper import multi_to_uni
from matplotlib import pyplot as plt
import sklearn as sk
import seaborn as sns

def obtain_na_frequencies(df_multi_strength):
	df_multi_strength_measures = df_multi_strength[machine_measures]
	return df_multi_strength_measures.isnull().sum() / len(df_multi_strength_measures)

def obtain_column_means(df_multi_strength):
	return df_multi_strength[machine_measures].mean()

def obtain_global_mean(df_multi_strength):
	return df_multi_strength[machine_measures].sum().sum() / df_multi_strength[machine_measures].count().sum()

def perform_KDE_density_estimations(df_uni_strength):
	for machine_measure in machine_measures:
		df_uni_strength_by_machine = df_uni_strength[df_uni_strength.Machine == machine_measure]

		_, ax = plt.subplots(figsize=(8, 6))
		ax = sns.kdeplot(df_uni_strength_by_machine['value'], shade=True, color='r', bw='silverman')
		plt.savefig('kdf_plot_' + machine_measure)

def print_data_properties():
	df_multi_strength = get_all_multi_strengths(import_data('strength_2016'))
	df_multi_strength, _ = get_multi_strengths(df_multi_strength, months=6)

	print('Shape: ', df_multi_strength.shape)

	nan_frequencies = obtain_na_frequencies(df_multi_strength)
	print(('NaN Frequencies\n{}').format(nan_frequencies))

	means = obtain_column_means(df_multi_strength)
	print('Column means', means)

	global_mean = obtain_global_mean(df_multi_strength)
	print('Global mean', global_mean)

	perform_KDE_density_estimations(multi_to_uni(df_multi_strength))


print_data_properties()
