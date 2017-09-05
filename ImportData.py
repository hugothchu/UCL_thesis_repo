import pandas as pd
import numpy as np

PARENT_PATH = '/home/hugo/Documents/Thesis-Private/'
BATCH_PATH_2013 = 'provided-by-eGym_2015-11-12/'
BATCH_PATH_2016 = '2016_Data/'

analysis = {
    'strength_2016' : PARENT_PATH + BATCH_PATH_2016 + 'Strength_measurement_data.csv',
    'body_data_2016': PARENT_PATH + BATCH_PATH_2016 + 'Body_Data.csv',
    'blood_pressure_2016' : PARENT_PATH + BATCH_PATH_2016 + 'Blood_Pressure.csv',
    'coordination_2016' : PARENT_PATH + BATCH_PATH_2016 + 'Coordination_Test.csv',
    'mobility_2016' : PARENT_PATH + BATCH_PATH_2016 + 'Flexibility_Screen.csv',
    'anamnesis_2016' : PARENT_PATH + BATCH_PATH_2016 + 'Trainer_App_Anamnesis.csv',
    'polar_2016' : PARENT_PATH + BATCH_PATH_2016 + 'Polar_Test.csv',
    'pwc_2016' : PARENT_PATH + BATCH_PATH_2016 + 'PWC_Test.csv',
    
    'strength_2013' : PARENT_PATH + BATCH_PATH_2013 + 'Strength_measurement_data_starting_2013.csv',
    'blood_pressure_2013' : PARENT_PATH + BATCH_PATH_2013 + 'Blood_Pressure_starting_2013.csv',
    'coordination_2013' : PARENT_PATH + BATCH_PATH_2013 + 'Coordination_Test_starting_2013.csv',
    'mobility_2013' : PARENT_PATH + BATCH_PATH_2013 + 'Mobility_Screen_starting_2013.csv',
    'anamnesis_2013' : PARENT_PATH + BATCH_PATH_2013 + 'Further_Anamnesis_Data_starting_2013.csv',
    'polar_2013' : PARENT_PATH + BATCH_PATH_2013 + 'Polar_Test_starting_2013.csv',
    'pwc_2013' : PARENT_PATH + BATCH_PATH_2013 + 'PWC_Tests_starting_2013.csv'
}

def import_data(analysis_type):
    pickle_path = PARENT_PATH + analysis_type
    try:
        df = pd.read_pickle(pickle_path)
    except FileNotFoundError:
        file_path = analysis.get(analysis_type)
        df = pd.read_csv(file_path, sep='\t')
        df.replace('(null)', np.nan, inplace=True)
        df.to_pickle(pickle_path)
    return df
