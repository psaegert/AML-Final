
import pandas as pd
from tqdm.auto import tqdm

print('Reading Data...')
# all necessary data has to be in a dedicated 'data' folder
data = pd.read_csv('data/data_geo_hosp.csv')

# save data for later sanity check
data_length = len(data)
education = pd.read_csv('data/county_data/Education.csv', encoding = "latin")
population = pd.read_csv('data/county_data/PopulationEstimates.csv', encoding = "latin")
unemployment = pd.read_csv('data/county_data/Unemployment.csv', encoding = "latin")
poverty = pd.read_csv('data/county_data/PovertyEstimates.csv', encoding = "latin")

print('Sorting Data by FIPS Code...')
# sort data
data = data.sort_values("county_fips_code")
education = education.sort_values("FIPS Code")
population = population.sort_values("FIPStxt")
unemployment = unemployment.sort_values("FIPS_Code")
poverty = poverty.sort_values("FIPStxt")

def fips_lookup(fips_column, current_index, fips, data):
    for idx, fipsi in enumerate(fips_column[current_index:]):
        if fipsi == fips:
            return data[idx+current_index]
    print("not found")
    return "NaN"

def fips_lookup2(column, fips, type_column, type_name, data):
    for idx, fipsi in enumerate(column):
        if (fipsi == fips):
            for i in range(len(type_column[idx:])):
                if type_column[idx+i] == type_name:
                    return data[idx+i]
    print("not found")
    return "NaN"


print('Combining...')
# perform combination of data sets (takes ~1 minute on Intel® Core™ i5-5300U CPU @ 2.30GHz × 4 )
current_fips = data["county_fips_code"].iloc[0]
old_fips = 0
current_index = 0
pop1_list = [None] * len(data)
pop2_list = [None] * len(data)
pop3_list = [None] * len(data)
pop4_list = [None] * len(data)
edu1_list = [None] * len(data)
edu2_list = [None] * len(data)
edu3_list = [None] * len(data)
edu4_list = [None] * len(data)
empl1_list = [None] * len(data)
empl2_list = [None] * len(data)
empl3_list = [None] * len(data)
pov1_list = [None] * len(data)

for idx, fips in enumerate(tqdm(data["county_fips_code"])):
    current_fips = fips
    
    # whenever fips changes, we cache the necessary census data
    if (current_fips != old_fips):
        
        # print current_fips to keep track of progress
        pop1 = fips_lookup(population["FIPStxt"], current_index, fips, population["Rural-urban_Continuum Code_2013"])
        pop2 = fips_lookup(population["FIPStxt"], current_index, fips, population["Urban_Influence_Code_2013"])
        pop3 = fips_lookup(population["FIPStxt"], current_index, fips, population["Economic_typology_2015"])
        pop4 = fips_lookup(population["FIPStxt"], current_index, fips, population["POP_ESTIMATE_2019"])
        
        edu1 = fips_lookup(education["FIPS Code"], current_index, fips, education["Percent of adults with less than a high school diploma, 2015-19"])
        edu2 = fips_lookup(education["FIPS Code"], current_index, fips, education["Percent of adults with a high school diploma only, 2015-19"])
        edu3 = fips_lookup(education["FIPS Code"], current_index, fips, education["Percent of adults completing some college or associate's degree, 2015-19"])
        edu4 = fips_lookup(education["FIPS Code"], current_index, fips, education["Percent of adults with a bachelor's degree or higher, 2015-19"])
        
        empl1 = fips_lookup2(unemployment["FIPS_Code"], fips, unemployment["Attribute"], "Unemployment_rate_2020", unemployment["Value"])
        empl2 = fips_lookup2(unemployment["FIPS_Code"], fips, unemployment["Attribute"], "Median_Household_Income_2019", unemployment["Value"])
        empl3 = fips_lookup2(unemployment["FIPS_Code"], fips, unemployment["Attribute"], "Med_HH_Income_Percent_of_State_Total_2019", unemployment["Value"])
       
        pov1 = fips_lookup2(poverty["FIPStxt"], fips, poverty["Attribute"], "PCTPOVALL_2019", poverty["Value"])
        current_index += 1
    
    pop1_list[idx] = pop1
    pop2_list[idx] = pop2
    pop3_list[idx] = pop3
    pop4_list[idx] = pop4
    edu1_list[idx] = edu1
    edu2_list[idx] = edu2
    edu3_list[idx] = edu3
    edu4_list[idx] = edu4
    empl1_list[idx] = empl1
    empl2_list[idx] = empl2
    empl3_list[idx] = empl3
    pov1_list[idx] = pov1

    old_fips = current_fips


print('Finalizing...')
data["rural-urban_Continuum Code_2013"] = pop1_list
data["urban_influence_code_2013"] = pop2_list
data["economic_typology_2015"] = pop3_list
data["pop_estimate_2019"] = pop4_list
data["less_hs_pct"] = edu1_list
data["hs_only_pct"] = edu2_list
data["some_college_pct"] = edu3_list
data["bachelor_higher_pct"] = edu4_list
data["unempl_rate"] = empl1_list
data["med_hh_income_19"] = empl2_list
data["med_hh_income_pct_state_total_19"] = empl3_list
data["poverty_pct_19"] = pov1_list

print('Performing Sanity Check...')
#lastly, perform sanity check
if (data_length == len(data)):
    data.to_csv('data.csv', index=False)
else:
    print("size of data sets is different")