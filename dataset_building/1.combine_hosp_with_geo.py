import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from tqdm import tqdm

df = pd.read_csv('../data/hosp_data/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility.csv')

print(df.info())

print('Omitting unnecessary columns and rename...')
df = df[['collection_week', 'fips_code', 'inpatient_beds_7_day_avg', 'inpatient_beds_used_7_day_avg', 'total_icu_beds_7_day_avg', 'icu_beds_used_7_day_avg' ]]
df = df.rename(columns={'collection_week': 'week', 'fips_code': 'fips_code', 'inpatient_beds_7_day_avg': 'inp_beds', 'inpatient_beds_used_7_day_avg': 'inp_beds_used', 'total_icu_beds_7_day_avg': 'icu_beds', 'icu_beds_used_7_day_avg': 'icu_beds_used'})

print('Filtering NaN values...')
df = df.dropna()

# filter rows with -999999.0 values (not enough data to build 7 day average)
df = df[df.inp_beds != -999999.0]
df = df[df.inp_beds_used != -999999.0]
df = df[df.icu_beds != -999999.0]
df = df[df.icu_beds_used != -999999.0]

# now we create a county sum (takes ~ 20 min on google colab)
print('Creating Country Sum...')
county_data = {'week': [], 'fips_code': [], 'inp_beds': [], 'inp_beds_used': [], 'icu_beds': [], 'icu_beds_used': []}

for week in tqdm(df.week.unique()):
  for fips in df.loc[df.week == week, 'fips_code'].unique():
    df_week_fips = df[(df.week == week) & (df.fips_code == fips)]

    county_data['week'].append(week)
    county_data['fips_code'].append(fips)
    county_data['inp_beds'].append(df_week_fips.inp_beds.sum())
    county_data['inp_beds_used'].append(df_week_fips.inp_beds_used.sum())
    county_data['icu_beds'].append(df_week_fips.icu_beds.sum())
    county_data['icu_beds_used'].append(df_week_fips.icu_beds_used.sum())

df = pd.DataFrame(county_data)

# calculate county ratios
print('Calculating county ratios...')
df['used_bed_ratio'] = df['inp_beds_used']/df['inp_beds']
df['used_icu_ratio'] = df['icu_beds_used']/df['icu_beds']
df = df.drop(columns = ['inp_beds', 'inp_beds_used', 'icu_beds', 'icu_beds_used'])

# 0 in the denominator creates NaN values. We exchange NaN values with a ratio of 1
df = df.fillna(1.)

# remove ratios bigger than 1 since we dont know how to interpret them
df = df[(df.used_bed_ratio <= 1) & (df.used_icu_ratio <= 1)]

# turn weekly format to monthly format
df['week'] = pd.to_datetime(df['week'], format = "%Y/%m/%d")
df['week'] = df['week'] + DateOffset(months=1)
df['week'] = df['week'].dt.strftime('%Y-%m')
df = df.rename(columns = {'week' : 'month'})

# average ratios over month (takes ~ 3 min on google colab)
print('Averaging ratios over month...')
monthly_data = {'month': [], 'fips_code': [], 'used_bed_ratio': [], 'used_icu_ratio': []}

for month in tqdm(df.month.unique()):
  for fips in df.loc[df.month == month, 'fips_code'].unique():
    df_month_fips = df[(df.month == month) & (df.fips_code == fips)]

    monthly_data['month'].append(month)
    monthly_data['fips_code'].append(fips)
    monthly_data['used_bed_ratio'].append(df_month_fips.used_bed_ratio.mean())
    monthly_data['used_icu_ratio'].append(df_month_fips.used_icu_ratio.mean())

df = pd.DataFrame(monthly_data)
df['fips_code'] = df['fips_code'].astype(int)

# loading data
df_hos = df
df_geo = pd.read_csv('../data/partially_combined/data_geo.csv')

# mapping month and fips code in the geo dataset to used_bed_ratio and used_icu_ratio in the hospital dataset
# first we create a dictionary to encode the mapping
print('Creating Dicts...')
mapping_bed = {}
mapping_icu = {} 

for month in tqdm(df_geo.case_month.unique()):
  mapping_bed[month] = {}
  mapping_icu[month] = {}

  for fips in df_geo[df_geo['case_month'] == month].county_fips_code.unique():
    used_bed_ratio = df_hos[(df_hos.month == month) & (df_hos.fips_code == fips)].used_bed_ratio.values
    used_icu_ratio = df_hos[(df_hos.month == month) & (df_hos.fips_code == fips)].used_icu_ratio.values

    if (len(used_bed_ratio) == 0) or (len(used_icu_ratio) == 0):
      mapping_bed[month][fips] = np.nan
      mapping_icu[month][fips] = np.nan

    else:
      mapping_bed[month][fips] = used_bed_ratio[0]
      mapping_icu[month][fips] = used_icu_ratio[0]

# now we use the dictionary for our mapping (takes ~ 2 min on google colab)
print('Mapping fips code and month...')
f_bed = lambda x: mapping_bed[x.case_month][x.county_fips_code]
f_icu = lambda x: mapping_icu[x.case_month][x.county_fips_code]

df_geo['used_bed_ratio'] = df_geo.loc[:, ['case_month', 'county_fips_code']].apply(f_bed, axis = 1)
df_geo['used_icu_ratio'] = df_geo.loc[:, ['case_month', 'county_fips_code']].apply(f_icu, axis = 1)

# drop values without ratios
df_geo = df_geo.dropna()

# save data
print('Writing Geo_Hosp Dataset...')
df_geo.to_csv('../data/partially_combined/data_geo_hosp.csv', index=False)
