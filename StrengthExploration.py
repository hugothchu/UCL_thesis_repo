import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, pickle
from ImportData import import_data

def get_strengths(df_strength, months):
    pickle_path = 'strengths_' + str(months)

    if os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as handle:
            stored_strengths = pickle.load(handle)
            df_strength_med_agg = stored_strengths['df_strength_med_agg']
            df_strength_max_agg = stored_strengths['df_strength_max_agg']
    else:
        df_strength = process_timestamp_and_clean_data(df_strength)

        # Calculate time(months) since first workout
        #df_first_workout = df_strength.groupby(['machineTypeProduction', 'id'])['timestamp'].min().rename('first_workout').to_frame().reset_index()
        df_first_workout = df_strength.groupby(['machineTypeProduction', 'user_id'])['timestamp'].min().rename('first_workout').to_frame().reset_index()

        df_strength = pd.merge(df_strength, df_first_workout, how='inner', on=['machineTypeProduction', 'user_id'])
        df_strength['months_since_first_workout'] = pd.to_datetime(df_strength['timestamp'] - df_strength['first_workout'])
        df_strength['months_since_first_workout'] = df_strength['months_since_first_workout'].map(lambda x: 12 * (x.year - 1970) + x.month - 1)

        df_strength = df_strength.drop('first_workout', 1)

        # Filter for athletes who trained at least once for all 6 consecutive months since joining
        df_strength = df_strength[df_strength['months_since_first_workout'] <= months - 1]
        unique_months = df_strength.groupby(['machineTypeProduction', 'user_id'])['months_since_first_workout'].nunique().rename('unique_months').to_frame().reset_index()

        unique_months = unique_months[unique_months['unique_months'] >= months]

        df_strength = pd.merge(df_strength, unique_months, how='inner', on=['machineTypeProduction', 'user_id'])
        df_strength  = df_strength.drop('unique_months', 1)
        df_strength  = df_strength.drop('timestamp', 1)

        # Calculate max/median strength for each month
        df_strength_med_agg = df_strength.groupby(['machineTypeProduction', 'user_id', 'months_since_first_workout'])['value'].median().to_frame().reset_index()
        df_strength_max_agg = df_strength.groupby(['machineTypeProduction', 'user_id', 'months_since_first_workout'])['value'].max().to_frame().reset_index()

        # Convert Machine Type to one-hot
        machine_one_hot = pd.get_dummies(df_strength_med_agg['machineTypeProduction'])
        df_strength_med_agg = df_strength_med_agg.join(machine_one_hot)

        machine_one_hot = pd.get_dummies(df_strength_max_agg['machineTypeProduction'])
        df_strength_max_agg = df_strength_max_agg.join(machine_one_hot)

        with open(pickle_path, "wb") as handle:
            pickle.dump({'df_strength_med_agg': df_strength_med_agg, 'df_strength_max_agg': df_strength_max_agg}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return df_strength_med_agg, df_strength_max_agg

def process_timestamp_and_clean_data(df_strength):
    df_strength['timestamp'] = pd.to_datetime(df_strength['timestamp'], infer_datetime_format=True)
    #df_strength['value'] = df_strength['value'].apply(lambda x:float(x.replace(',', '.')))
    df_strength = df_strength.drop(df_strength[df_strength.value == 0].index)
    return df_strength

def get_strengths_with_gender(df_strength, df_body):
    df_body['updated'] = df_body['updated'].fillna('1900-01-01 00:00:00')
    df_body['updated'] = pd.to_datetime(df_body['updated'], infer_datetime_format=True)
    df_body_last_entry = df_body.groupby(['User_ID'])['updated'].max().to_frame().reset_index()
    df_body = pd.merge(df_body, df_body_last_entry, how='inner', on=['updated', 'User_ID'])
    df_body = df_body.dropna(subset=['Gender'])

    merged_df = pd.merge(df_strength, df_body, how='left', left_on='user_id', right_on='User_ID')
    merged_df.drop(['User_ID', 'updated', 'yearOfBirth', 'bodyMuscleShare', 'weight', 'height', 'bodyFatShare', 'dataSource', 'gymLocation_id'], axis=1, inplace=True)
    return merged_df

def plot_strengths(df_strength):
    plt.hist(df_strength['value'], bins=25)
    plt.title("Strength Dist")
    plt.xlabel("Performance")
    plt.ylabel("Frequency")
    plt.show()
