import numpy as np
import pandas as pd

data = pd.read_csv('../../data/COVID-19_Case_Surveillance_Public_Use_Data_with_Geography.csv')


print(data.info())


data = data.drop(columns=['process', 'exposure_yn', 'icu_yn', 'case_positive_specimen_interval', 'underlying_conditions_yn', 'res_state', 'res_county'])
data = data.dropna()

data.current_status = data.current_status.map({'Laboratory-confirmed case': 1, 'Probable Case': 0}).astype(int)

data['sex_female'] = data.sex.map({'Female': 1, 'Male': 0, 'Missing': -1, 'Other': -1, 'Unknown': -1}).astype(int)
data = data.drop(columns=['sex'])

data.hosp_yn = data.hosp_yn.map({'No': 0, 'Yes': 1, 'Missing': -1, 'Unknown': -1}).astype(int)

data.death_yn = data.death_yn.map({'No': 0, 'Yes': 1, 'Missing': -1, 'Unknown': -1}).astype(int)

data.symptom_status = data.symptom_status.map({'Symptomatic': 1, 'Missing': 0}).astype(int)

data.age_group = data.age_group.map({'0 - 17 years': 0, '18 to 49 years': 18, '50 to 64 years': 50, '65+ years': 65, 'Missing': 'missing'})

data = pd.concat([data, pd.get_dummies(data['age_group'], prefix='age')], axis=1)
data = data.drop(columns=['age_group'])

data['race_missing'] = np.logical_or(data.race.str.contains('Missing'), data.race.str.contains('Unknown'))
data['ethnicity_missing'] = np.logical_or(data.ethnicity.str.contains('Missing'), data.ethnicity.str.contains('Unknown'))

data['race_native'] = data.race.str.contains('American Indian/Alaska Native')
data['race_asian'] = data.race.str.contains('Asian')
data['race_black'] = data.race.str.contains('Black')
data['race_other'] = data.race.str.contains('Multiple/Other')
data['race_native_pacific'] = data.race.str.contains('Native Hawaiian/Other Pacific Islander')
data['race_white'] = data.race.str.contains('White')

data['ethnicity_hispanic'] = data.ethnicity.str.contains('Hispanic/Latino')

data = data.drop(columns=['race', 'ethnicity'])
drop_mask = (
    (data.current_status == 0)
    # Features
    | (data.sex_female == -1) | (data.age_missing == 1) | (data.race_missing)
    # Dependent Variables
    | (data.hosp_yn == -1) | (data.death_yn == -1)
)
data = data.loc[np.logical_not(drop_mask)]

data.iloc[:, 4:] = data.iloc[:, 4:].astype(bool)

data = data.reset_index()
data = data.drop(columns=['index', 'age_missing', 'race_missing', 'ethnicity_missing', 'current_status'])

data.state_fips_code = data.state_fips_code.astype(int)
data.county_fips_code = data.county_fips_code.astype(int)
data.case_onset_interval = data.case_onset_interval.astype(int)

data = data[list(data.columns.values)[5:] + list(data.columns.values[:5])]

data.to_csv('../../data/data_geo.csv', index=False)