from StrengthExploration import process_timestamp_and_clean_data
from pandas import Grouper
import numpy as np
import pandas as pd
import os, pickle
from ImportData import import_data
from datetime import datetime

machine_measures = [
    'M1_value', 'M2_value', 'M3_value', 'M4_value', 'M5_value', 'M6_value', 'M7_value', 'M8_value', 'M8_value_2', 'M9_value', 
    'M10_value', 'M11_value', 'M12_value', 'M12_value_2', 'M13_value', 'M14_value', 'M15_value', 'M16_value', 'M17_value', 'M18_value'
]

def get_all_multi_strengths(df_strength):
    pickle_path_str_agg_all = 'all_strengths'
    
    if os.path.isfile(pickle_path_str_agg_all):
        with open(pickle_path_str_agg_all, 'rb') as handle:
            df_multi_strength = pickle.load(handle)
    else:
        df_strength = process_timestamp_and_clean_data(df_strength)
        df_by_user_date = df_strength.groupby(['user_id', Grouper(key='timestamp', freq='D')])
        
        df_multi_strength = df_by_user_date.apply(transform_to_multi_strength)
        df_multi_strength = df_multi_strength.drop_duplicates()

        with open(pickle_path_str_agg_all, "wb") as handle:
            pickle.dump(df_multi_strength, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return df_multi_strength

def get_multi_strengths(df_strength, months):
    pickle_path_str_agg_by_month = 'all_strengths_' + str(months)

    if os.path.isfile(pickle_path_str_agg_by_month):
        with open(pickle_path_str_agg_by_month, 'rb') as handle:
            stored_strengths = pickle.load(handle)
            df_strength_med_agg = stored_strengths['df_strength_med_agg']
            df_strength_max_agg = stored_strengths['df_strength_max_agg']

    else:
        df_multi_strength = get_all_multi_strengths(df_strength)
            
        df_first_workout = df_multi_strength.groupby('User_ID')['Date'].min().rename('First_Workout').to_frame().reset_index()
        df_strength = pd.merge(df_multi_strength, df_first_workout, how='inner', on='User_ID')
        df_strength['Months_Since_First_Workout'] = pd.to_datetime(df_strength['Date'] - df_strength['First_Workout'])
        df_strength['Months_Since_First_Workout'] = df_strength['Months_Since_First_Workout'].map(lambda x: 12 * (x.year - 1970) + x.month - 1)
        
        df_strength = df_strength.drop('First_Workout', 1)

        # Filter for athletes who trained at least once for all 6 consecutive months since joining
        df_strength = df_strength[df_strength['Months_Since_First_Workout'] <= months - 1]
        unique_months = df_strength.groupby('User_ID')['Months_Since_First_Workout'].nunique().rename('Unique_Months').to_frame().reset_index()
        unique_months = unique_months[unique_months['Unique_Months'] >= months]

        df_strength = pd.merge(df_strength, unique_months, how='inner', on='User_ID')
        df_strength = df_strength.drop('Unique_Months', 1)
        df_strength = df_strength.drop('Date', 1)

        # Calculate Max/Median Aggregate per month
        df_strength_med_agg = df_strength.groupby(['User_ID', 'Months_Since_First_Workout'])[machine_measures].median()
        df_strength_max_agg = df_strength.groupby(['User_ID', 'Months_Since_First_Workout'])[machine_measures].max()

        df_strength_med_agg.reset_index(inplace=True)
        df_strength_max_agg.reset_index(inplace=True)

        with open(pickle_path_str_agg_by_month, 'wb') as handle:
            pickle.dump({'df_strength_med_agg': df_strength_med_agg, 'df_strength_max_agg': df_strength_max_agg}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return df_strength_med_agg, df_strength_max_agg

def transform_to_multi_strength(group):
    return pd.DataFrame({
        'User_ID': group.iloc[-1].user_id,
        'Date': group.iloc[-1].timestamp,
        'M1_value': get_value_if_not_empty(group, 'M1', 1),
        'M2_value': get_value_if_not_empty(group, 'M2', 1),
        'M3_value': get_value_if_not_empty(group, 'M3', 1),
        'M4_value': get_value_if_not_empty(group, 'M4', 1),
        'M5_value': get_value_if_not_empty(group, 'M5', 1),
        'M6_value': get_value_if_not_empty(group, 'M6', 1),
        'M7_value': get_value_if_not_empty(group, 'M7', 1),
        'M8_value': get_value_if_not_empty(group, 'M8', 1),
        'M8_value_2': get_value_if_not_empty(group, 'M8', 2),
        'M9_value': get_value_if_not_empty(group, 'M9', 1),
        'M10_value': get_value_if_not_empty(group, 'M10', 1),
        'M11_value': get_value_if_not_empty(group, 'M11', 1),
        'M12_value': get_value_if_not_empty(group, 'M12', 1),
        'M12_value_2': get_value_if_not_empty(group, 'M12', 2),
        'M13_value': get_value_if_not_empty(group, 'M13', 1),
        'M14_value': get_value_if_not_empty(group, 'M14', 1),
        'M15_value': get_value_if_not_empty(group, 'M15', 1),
        'M16_value': get_value_if_not_empty(group, 'M16', 1),
        'M17_value': get_value_if_not_empty(group, 'M17', 1),
        'M18_value': get_value_if_not_empty(group, 'M18', 1)
    }, index=group.index)

def get_value_if_not_empty(df, machine_type, value_to_get):
    if not df[df.machineTypeProduction == machine_type].empty:
        if value_to_get == 1:
            return df[df.machineTypeProduction == machine_type].iloc[-1].value
        elif value_to_get == 2:
            return df[df.machineTypeProduction == machine_type].iloc[-1].second_Value
        else:
            return np.nan
    else:
        return np.nan

def get_strengths_with_gender(df_strength, df_body):
    df_body['updated'] = df_body['updated'].fillna('1900-01-01 00:00:00')
    df_body['updated'] = pd.to_datetime(df_body['updated'], infer_datetime_format=True)
    df_body_last_entry = df_body.groupby(['User_ID'])['updated'].max().to_frame().reset_index()
    df_body = pd.merge(df_body, df_body_last_entry, how='inner', on=['updated', 'User_ID'])
    df_body = df_body.dropna(subset=['Gender'])

    merged_df = pd.merge(df_strength, df_body, how='inner', left_on='User_ID', right_on='User_ID')
    merged_df.drop(['User_ID', 'updated', 'yearOfBirth', 'bodyMuscleShare', 'weight', 'height', 'bodyFatShare', 'dataSource', 'gymLocation_id'], axis=1, inplace=True)
    return merged_df
    